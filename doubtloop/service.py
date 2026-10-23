"""Thin service layer used by the Streamlit UI and the evaluation scripts."""
import json
import shutil

from langgraph.types import Command

from . import config, knowledge, store
from .graph import get_app


def startup(seed: bool = True) -> dict:
    """Create tables, build the notes index, load assignments, and seed demo data once."""
    store.init()
    chunks = knowledge.build_notes_index()
    knowledge.load_assignments()
    if seed and store.count() == 0:
        load_seed()
    return {"chunks": chunks, "ta_memory": knowledge.ta_memory_count(), "doubts": store.count()}


def _cfg(doubt_id):
    return {"configurable": {"thread_id": doubt_id}}


def ask(student: str, question: str, created_at: str = None, chat_id: str = None,
        scope: str = "", auto_confirm: bool = False, on_step=None, subject: str = None,
        style: str = "detailed", on_token=None) -> dict:
    """Run one doubt through the graph. Returns the stored doubt record.

    chat_id      conversation thread (conversation memory is per chat)
    scope        course file to focus on ('' = all materials)
    auto_confirm skip the student's 'send to TA?' approval (used by evaluation scripts)
    on_step      callback(node_name) called as each agent step finishes (live progress in the UI)
    subject      which subject's library to answer from (default: COURSE_NAME in .env)
    style        answer length: brief | detailed | exam
    """
    doubt_id = store.new_id()
    inputs = {"doubt_id": doubt_id, "student": student, "raw_question": question,
              "chat_id": chat_id, "scope": scope or "", "created_at": created_at or store.now(),
              "subject": subject or knowledge.DEFAULT_SUBJECT, "style": style or "detailed"}
    if on_step is None and on_token is None:
        get_app().invoke(inputs, _cfg(doubt_id))
    else:   # stream node-by-node (live progress) and token-by-token (answer appears as it is written)
        for mode, chunk in get_app().stream(inputs, _cfg(doubt_id), stream_mode=["updates", "custom"]):
            if mode == "custom" and on_token and isinstance(chunk, dict) and "token" in chunk:
                on_token(chunk["token"])
            elif mode == "updates" and on_step:
                for node in chunk:
                    if not node.startswith("__"):
                        on_step(node)
    d = store.get(doubt_id)
    if auto_confirm and d["status"] == "awaiting_student":
        d = student_confirm(doubt_id, send=True)
    return d


def student_confirm(doubt_id: str, send: bool, note: str = "") -> dict:
    """Resume the graph after the student decides whether to send the doubt to a TA."""
    app = get_app()
    snap = app.get_state(_cfg(doubt_id))
    if snap and snap.next:
        app.invoke(Command(resume={"send": send, "note": note}), _cfg(doubt_id))
    else:                                         # no live checkpoint (e.g. after a reset)
        store.update(doubt_id, status="escalated" if send else "withdrawn")
    return store.get(doubt_id)


def ta_reply(doubt_id: str, answer: str, ta_name: str = "TA", save_to_memory: bool = True) -> dict:
    """Resume the paused graph with the TA's answer (human-in-the-loop)."""
    app = get_app()
    snap = app.get_state(_cfg(doubt_id))
    if snap and snap.next:                       # graph is paused at wait_for_ta
        app.invoke(Command(resume={"answer": answer, "ta": ta_name, "save": save_to_memory}),
                   _cfg(doubt_id))
    else:                                        # e.g. seeded doubts with no live checkpoint
        d = store.get(doubt_id)
        if save_to_memory:
            knowledge.add_ta_answer(d.get("standalone") or d["question"], answer, ta_name, d["topic"], doubt_id,
                                    subject=d.get("subject") or None)
        info = "TA answer stored in vector memory" if save_to_memory else "TA replied to this student only"
        trace = d["trace"] + [{"node": "learn", "info": info, "ms": 0}]
        store.update(doubt_id, status="ta_answered", answer=answer, answered_by=f"ta:{ta_name}",
                     resolved_at=store.now(), trace=trace)
    result = store.get(doubt_id)
    result["auto_resolved"] = resolve_similar_pending(doubt_id) if save_to_memory else []
    return result


def resolve_similar_pending(doubt_id: str) -> list:
    """One TA answer can clear several waiting doubts that ask the same thing."""
    from .embeddings import thresholds

    src = store.get(doubt_id)
    resolved = []
    subject = src.get("subject") or knowledge.DEFAULT_SUBJECT
    for d in store.list_doubts(status=["escalated"], subject=subject):
        hits = knowledge.search_ta_memory(d.get("standalone") or d["question"], subject=subject)
        if hits and hits[0]["id"] == doubt_id and hits[0]["sim"] >= thresholds()["ta"]:
            trace = d["trace"] + [{"node": "ta_memory", "info": f"auto-resolved by TA answer to "
                                   f"{doubt_id} (sim {hits[0]['sim']:.2f})", "ms": 0}]
            store.update(d["id"], status="verified", answer=src["answer"],
                         answered_by=f"memory:{src['answered_by'].split(':', 1)[-1]}",
                         confidence=hits[0]["sim"], resolved_at=store.now(), trace=trace)
            resolved.append(d["id"])
    return resolved


def web_answer(doubt_id: str) -> dict:
    """Student approved a web search for this doubt (human-in-the-loop approval of a tool call)."""
    from . import agents, guardrails

    d = store.get(doubt_id)
    subject = d.get("subject") or knowledge.DEFAULT_SUBJECT
    if d.get("category") in ("assignment", "admin", "off_topic", "blocked"):
        raise ValueError("Web search isn't available for this kind of question.")
    if not knowledge.subject_info(subject).get("web", True):
        raise ValueError("Your teacher has turned off web answers for this class.")
    notes = "\n\n".join(s.get("text", "") for s in (d.get("sources") or []) if s.get("sim", 0) > 0)
    res = agents.web_answer(d.get("standalone") or d["question"], notes, subject)
    text, _ = guardrails.mask_pii(res["text"])
    how = (f"Google Search grounding, {len(res['sources'])} sources" if res["mode"] == "web"
           else "search unavailable, used Gemini's general knowledge")
    trace = (d.get("trace") or []) + [{"node": "web_agent", "info": f"student approved; {how}", "ms": 0}]
    store.update(doubt_id, web_answer=text, web_sources=res["sources"], trace=trace)
    return store.get(doubt_id)


def double_check(doubt_id: str) -> dict:
    """On-demand reflection: the AI critic re-reads the answer against the notes it cited."""
    from . import agents

    d = store.get(doubt_id)
    chunks = [{"text": s.get("text", ""), "meta": {"unit": s.get("unit", ""), "section": s.get("section", "")}}
              for s in (d.get("sources") or []) if s.get("text")]
    c = agents.critique(d.get("standalone") or d["question"], d.get("answer") or "", chunks)
    verdict = ("supported by the notes" if c["grounded"] else "may not be fully supported") + \
        f" (score {c['score']:.2f})" + (f": {c['issues']}" if c["issues"] else "")
    trace = (d.get("trace") or []) + [{"node": "double_check", "info": verdict, "ms": 0}]
    store.update(doubt_id, trace=trace)
    return store.get(doubt_id)


def ta_dismiss(doubt_id: str, reason: str, ta_name: str = "TA"):
    store.update(doubt_id, status="refused", answer=reason, answered_by=f"ta:{ta_name}",
                 resolved_at=store.now())


SEED_SUBJECT = "Agentic AI (Sem VII)"      # the demo data is about the Agentic AI course


def load_seed():
    """Synthetic demo history (clearly flagged is_seed=1) so the dashboard isn't empty."""
    for item in json.loads((config.SEED_DIR / "ta_answers.json").read_text(encoding="utf-8")):
        knowledge.add_ta_answer(item["question"], item["answer"], item["ta"], item["topic"], item["id"],
                                subject=SEED_SUBJECT)
    for d in json.loads((config.SEED_DIR / "doubts.json").read_text(encoding="utf-8")):
        d.setdefault("guardrails", [])
        d.setdefault("trace", [])
        d.setdefault("sources", [])
        d["is_seed"] = 1
        d["subject"] = SEED_SUBJECT
        store.save(d)


def reset_all(seed: bool = True):
    """Delete all doubts, TA memory and checkpoints (course notes index is kept)."""
    store.clear_all()
    knowledge.reset_ta_memory()
    # Old checkpoints are harmless: every doubt gets a fresh thread id.
    if seed:
        load_seed()


def wipe_runtime():
    shutil.rmtree(config.RUNTIME_DIR, ignore_errors=True)