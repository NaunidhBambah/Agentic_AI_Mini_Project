"""The LangGraph workflow (orchestrator) that connects guardrails, agents and the human TA.

input_guard -> contextualize (conversation memory) -> retrieve -> router -> {ta_memory | answer(hint) | refuse | notify_ta}
ta_memory  -> hit: finalize            miss: answer
answer     -> critic -> (retry answer once) -> confidence_gate
confidence_gate -> output_guard (confident) | notify_ta (not confident)
output_guard -> finalize (ok) | notify_ta (guardrail withheld answer)
notify_ta -> wait_for_ta  [interrupt(): graph PAUSES until a TA answers]
wait_for_ta -> learn (save TA answer to vector memory) -> finalize
"""
import operator
import re
import sqlite3
import time
from functools import lru_cache, wraps
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import agents, config, guardrails, knowledge, store
from .embeddings import get_embedding, thresholds


class DoubtState(TypedDict, total=False):
    doubt_id: str
    student: str
    raw_question: str
    question: str
    standalone: str
    chat_id: str
    scope: str             # file the student focused on ('' = all materials)
    subject: str           # which subject's library to use
    style: str             # answer length: brief | detailed | exam
    scope_note: str        # set when the focused file had nothing and all materials were searched
    conversation: str      # last turns of this chat, for the answer agent
    student_note: str
    created_at: str
    block_reason: str
    category: str
    topic: str
    route_reason: str
    chunks: list
    top_sim: float
    first_top: float       # best match before the query was rewritten (used for off-topic decisions)
    assignment_hit: dict
    memory_hit: dict
    draft: str
    critique: dict
    retries: int
    confidence: float
    mode: str              # notes | hint | memory | ta | refused | blocked
    status: str
    final_answer: str
    answered_by: str
    output_ok: bool
    ta_answer: str
    ta_name: str
    ta_save: bool
    guard_events: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]


def traced(name):
    """Decorator: time each node and append a step to the trace (observability)."""
    def deco(fn):
        @wraps(fn)
        def inner(state):
            t0 = time.perf_counter()
            out = fn(state) or {}
            info = out.pop("_info", "")
            out["trace"] = [{"node": name, "info": info, "ms": int((time.perf_counter() - t0) * 1000)}]
            return out
        return inner
    return deco


# ------------------------------------------------------------------ nodes
@traced("input_guard")
def input_guard(state):
    text, events, block = guardrails.check_input(state["raw_question"])
    out = {"question": text, "guard_events": events, "retries": 0}
    if block:
        out.update(block_reason=block, mode="blocked", status="blocked", category="blocked",
                   topic="Blocked", final_answer=block, answered_by="guardrail")
        out["_info"] = f"BLOCKED: {events[-1]['guardrail'] if events else 'empty'}"
    else:
        out["_info"] = f"passed ({len(events)} PII masks)" if events else "passed"
    return out


@traced("contextualize")
def contextualize(state):
    """Conversation memory: read this student's earlier doubts (episodic memory in SQLite)."""
    past = [d for d in reversed(store.list_doubts(student=state["student"], limit=50,
                                                  chat_id=state.get("chat_id") or None))
            if d["status"] not in ("blocked", "withdrawn") and d["id"] != state["doubt_id"]]
    q = state["question"]
    if agents.is_history_question(q):
        return {"standalone": q, "category": "history", "topic": "Doubt history", "mode": "history",
                "status": "history", "final_answer": agents.history_answer(past),
                "answered_by": "memory:history", "confidence": 1.0,
                "_info": f"history question, answered from {len(past)} past doubts"}
    res = agents.contextualize(q, past)
    changed = res["standalone"] != q
    return {"standalone": res["standalone"], "conversation": agents.format_turns(past),
            "_info": (f"follow-up rewritten via {res['by']}: '{res['standalone'][:70]}'" if changed
                      else f"standalone ({len(past)} earlier doubts in this chat)")}


def _writer():
    """LangGraph's stream writer (sends answer tokens to the UI while the node runs); None outside streams."""
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except Exception:
        return None


def _subject(state) -> str:
    return state.get("subject") or knowledge.DEFAULT_SUBJECT


_PII_PHRASE = re.compile(r"\b(my|our)?\s*(phone|mobile|contact|email|e-mail|id|roll)( number| no\.?| id)?\s*(is|:)?\s*"
                         r"\[[A-Z_ ]+MASKED\][,.;:]?\s*", re.I)


def _search_text(q: str) -> str:
    """The question without masked personal data, so '[PHONE MASKED]' doesn't disturb the search."""
    q = _PII_PHRASE.sub(" ", q)
    return re.sub(r"\[[A-Z_ ]+MASKED\]", " ", q).strip() or q


def _q(state):
    return state.get("standalone") or state["question"]


@traced("retrieve")
def retrieve(state):
    q = _search_text(_q(state))
    th = thresholds()
    scope, subject = state.get("scope", ""), state.get("subject") or None
    chunks = knowledge.search_notes(q, source=scope, subject=subject)
    subs = agents.sub_queries(q)
    if subs:                                   # comparison: make sure every compared idea is retrieved
        seen = {c["id"] for c in chunks}
        for sq in subs:
            for c in knowledge.search_notes(sq, k=3, source=scope, subject=subject):
                if c["id"] not in seen:
                    chunks.append(c)
                    seen.add(c["id"])
    note, widened = "", ""
    top_scoped = max((c["sim"] for c in chunks), default=0.0)
    if scope and top_scoped < th["retrieval"] + 0.2 and not agents.is_summary_request(q):
        # Little or nothing in the focused file: check the whole library. Switch when the focused file has
        # nothing, or when another file matches clearly better (e.g. the topic is in another unit).
        wide = knowledge.search_notes(q, subject=subject)
        top_wide = max((c["sim"] for c in wide), default=0.0)
        better = top_wide > top_scoped + (0.05 if top_scoped < th["retrieval"] else 0.2)
        if top_wide >= th["retrieval"] + 0.05 and better:
            files = list(dict.fromkeys(knowledge.scope_label(c["meta"]["source"]) for c in wide))
            note = (f"Not found clearly in **{knowledge.scope_label(scope)}**, so I searched all course "
                    f"materials and found it in **{files[0]}**.")
            chunks, widened = wide, f"; focused file had nothing (top {top_scoped:.2f}), widened to all files"
    a_hits = knowledge.search_assignments(q)
    # Graded work = similar to a known assignment AND asks for a solution, or a near copy-paste of it.
    a_hit = {}
    if a_hits:
        sim = a_hits[0]["sim"]
        if (sim >= th["assignment"] and guardrails.is_task_request(q)) or sim >= min(0.9, th["assignment"] + 0.2):
            a_hit = a_hits[0]
    if not a_hit and guardrails.is_solution_request(q):
        a_hit = {"meta": {"title": "request for a complete solution"}, "sim": 1.0}
    top = max((c["sim"] for c in chunks), default=0.0)
    rewrote, first_top = "", top
    if top < th["retrieval"] + 0.1 and not agents.is_summary_request(q) and not a_hit:
        # Corrective retrieval: the first search was weak, so the agent rewrites the query and tries again
        for alt in agents.rewrite_queries(q, knowledge.library_topics(subject=subject)):
            again = knowledge.search_notes(alt, source=scope if not note else "", subject=subject)
            best = max((c["sim"] for c in again), default=0.0)
            if best > top + 0.05:
                chunks, top, rewrote = again, best, alt
        if rewrote:
            rewrote = f"; weak first search, retried as '{rewrote}'"
    n_hits = len(chunks)
    chunks = knowledge.with_neighbours(chunks)   # add the slides around each match (whole explanation)
    files = ", ".join(dict.fromkeys(c["meta"]["source"] for c in chunks))
    return {"chunks": chunks, "top_sim": top, "first_top": first_top, "assignment_hit": a_hit, "scope_note": note,
            "_info": f"{n_hits} matches + {len(chunks) - n_hits} neighbouring chunks from {files or 'no files'}, "
                     f"top sim {top:.2f}" + (f"; also searched {subs}" if subs else "") + widened + rewrote
                     + (f"; matches assignment '{a_hit['meta']['title']}' ({a_hit['sim']:.2f})" if a_hit else "")}


@traced("router")
def router(state):
    sections = [f"{c['meta']['unit']}: {c['meta']['section']} (sim {c['sim']:.2f})"
                for c in state.get("chunks", [])]
    # The ORIGINAL match decides whether this is about the course at all: a rewritten query can make even
    # "paneer recipe" look course-like, so it must not turn an off-topic message into a course doubt.
    r = agents.route(_q(state), min(state.get("top_sim", 0.0), state.get("first_top", 1.0)),
                     bool(state.get("assignment_hit")), thresholds()["retrieval"], sections,
                     scoped=bool(state.get("scope")),
                     library=knowledge.library_overview(subject=state.get("subject") or None),
                     lib_topics=knowledge.library_topics(subject=state.get("subject") or None))
    if r["topic"] in ("General", "") and r["category"] in ("concept", "summary", "assignment") \
            and state.get("top_sim", 0.0) < thresholds()["retrieval"]:
        r["topic"] = "Outside the notes"            # weak match: don't borrow the nearest slide's topic
    elif r["topic"] in ("General", "") and r["category"] in ("concept", "summary", "assignment"):
        chunks = [c for c in state.get("chunks", []) if c.get("sim", 0) > 0] or state.get("chunks", [])
        if chunks:
            src = chunks[0]["meta"]["source"]
            doc = next((d for d in knowledge.list_documents(state.get("subject") or None) if d["file"] == src), {})
            r["topic"] = agents.pick_topic(_q(state), chunks, doc.get("topics") or [])
    out = {"category": r["category"], "topic": r["topic"], "route_reason": r["reason"],
           "_info": f"{r['category']} / {r['topic']} ({r['by']})"}
    if r["category"] == "assignment":
        out["guard_events"] = [guardrails.event("academic_integrity",
                                                "graded question detected, switching to hint-only mode",
                                                "hint_mode")]
    return out


def _library_reply(state) -> str:
    """Answer 'can you see the notes / which files are uploaded?' from the library catalogue."""
    subject, scope = _subject(state), state.get("scope", "")
    docs = knowledge.list_documents(subject)
    if scope and not scope.startswith("unit:"):
        docs = [d for d in docs if d["file"] == scope] or docs
        intro = "Yes. You've selected this document, and I answer from it:"
    elif scope.startswith("unit:"):
        docs = [d for d in docs if d["unit"] == scope[5:]] or docs
        intro = f"Yes. You've selected {scope[5:]}, which has these documents:"
    else:
        intro = f"Yes. {subject} has {len(docs)} document{'s' if len(docs) != 1 else ''}, and I answer from all of them:"
    if not docs:
        return f"No documents have been uploaded for {subject} yet, so I can't answer from notes."
    lines = []
    for d in docs:
        line = f"- **{d['title']}** ({d['unit']}, {d['kind']}, {d['pages']} pages)"
        if d["summary"]:
            line += f": {d['summary']}"
        if d["topics"]:
            line += f"  \n  *Topics:* {', '.join(d['topics'][:6])}"
        lines.append(line)
    return intro + "\n\n" + "\n".join(lines) + ("\n\nAsk about any of these topics, or change **Focus on** "
                                                  "to search a different document.")


@traced("refuse")
def refuse(state):
    if state.get("category") == "library":
        return {"mode": "smalltalk", "status": "library", "answered_by": "agent", "confidence": 1.0,
                "final_answer": _library_reply(state), "_info": "described the available material"}
    if state.get("category") == "smalltalk":
        q = state["question"].lower()
        hello = next((f"Good {w}!" for w in ("morning", "afternoon", "evening", "night") if f"good {w}" in q),
                     "Hi!")
        if len(re.findall(r"[a-z]", q)) < 4:
            return {"mode": "smalltalk", "status": "smalltalk", "answered_by": "agent",
                    "final_answer": "That looks incomplete. Could you type your full doubt?",
                    "_info": "message too short"}
        if "how are you" in q:
            hello += " I'm doing well, thanks for asking."
        if re.search(r"\bthank", q):
            hello = "You're welcome!"
        return {"mode": "smalltalk", "status": "smalltalk", "answered_by": "agent",
                "final_answer": (f"{hello} I'm ready for your {_subject(state)} doubts. Ask about any "
                                 "topic, or ask me to summarise a unit or a document. Pick a file in "
                                 "**Focus on** to search only that file."),
                "_info": "greeting / chit-chat"}
    topics = knowledge.library_topics(subject=state.get("subject") or None)[:6]
    msg = (f"That's outside {_subject(state)}, so I'll skip it. Ask me anything from the uploaded "
           "course materials" + (f", for example: {', '.join(topics)}." if topics else "."))
    return {"mode": "refused", "status": "refused", "final_answer": msg, "answered_by": "guardrail",
            "guard_events": [guardrails.event("topic_filter", "off-topic question", "refused")],
            "_info": "off-topic refused"}


@traced("summarize")
def summarize(state):
    """Summary agent: reads the chosen document/unit in order (not just the 3 closest chunks)."""
    scope, subject = state.get("scope", ""), state.get("subject") or None
    m = re.search(r"\bunit[\s-]*(\d+)", _q(state), re.I)
    if not scope and m and any(d["unit"] == f"Unit {m.group(1)}" for d in knowledge.list_documents(subject)):
        scope = f"unit:Unit {m.group(1)}"           # "summarise unit 2" -> only Unit 2's files
    chunks = knowledge.document_chunks(scope, limit=24 if scope else 30, subject=subject)
    if not chunks:
        return {"draft": "NOT_IN_NOTES", "mode": "notes", "_info": "nothing to summarise"}
    writer = _writer()
    res = agents.summarize(_q(state), chunks, state.get("conversation", ""),
                           on_token=(lambda t: writer({"token": t})) if writer else None)
    files = sorted({c["meta"]["source"] for c in chunks})
    return {"chunks": chunks, "draft": res["text"], "mode": "summary", "confidence": 0.0,
            "_info": f"summarised {len(chunks)} chunks from {len(files)} file(s) via {res['by']}"}


@traced("ta_memory")
def ta_memory(state):
    hits = knowledge.search_ta_memory(_q(state), subject=state.get("subject") or None)
    if hits and hits[0]["sim"] >= thresholds()["ta"]:
        h = hits[0]
        return {"memory_hit": h, "mode": "memory", "status": "verified",
                "final_answer": h["meta"]["answer"], "confidence": h["sim"],
                "answered_by": f"memory:{h['meta']['ta']}",
                "_info": f"HIT: similar to '{h['text'][:60]}' (sim {h['sim']:.2f})"}
    best = f"best sim {hits[0]['sim']:.2f}" if hits else "memory empty"
    return {"memory_hit": {}, "_info": f"miss ({best})"}


@traced("answer_agent")
def answer_agent(state):
    hint = state.get("category") == "assignment"
    if not hint and state.get("top_sim", 0.0) < thresholds()["retrieval"]:
        # Nothing relevant was retrieved: don't spend 10-60 s asking the LLM, offer the TA straight away.
        return {"draft": "NOT_IN_NOTES", "mode": "notes",
                "_info": f"no relevant notes (top sim {state.get('top_sim', 0):.2f} < "
                         f"{thresholds()['retrieval']:.2f}), skipped the LLM"}
    feedback = state.get("critique", {}).get("issues", "") if state.get("retries") else ""
    writer = _writer()
    res = agents.answer(_q(state), state.get("chunks", []), hint=hint, feedback=feedback,
                        conversation=state.get("conversation", ""), subject=_subject(state),
                        style=state.get("style") or "detailed",
                        on_token=(lambda t: writer({"token": t})) if writer else None)
    return {"draft": res["text"], "mode": "hint" if hint else "notes",
            "_info": f"draft {len(res['text'])} chars via {res['by']}" + (" (retry)" if feedback else "")}


@traced("critic_agent")
def critic_agent(state):
    strong = state.get("top_sim", 0.0) >= thresholds()["retrieval"] + 0.2
    quick = config.CRITIC_MODE == "auto" and strong and not state.get("retries")
    if config.CRITIC_MODE == "off":
        c = agents.quick_check(state["draft"], state.get("chunks", []))
    else:
        c = agents.critique(_q(state), state["draft"], state.get("chunks", []), quick=quick)
    return {"critique": c, "retries": state.get("retries", 0) + (0 if c["grounded"] else 1),
            "_info": f"grounded={c['grounded']} score={c['score']:.2f} via {c['by']}"
                     + (f" | {c['issues'][:60]}" if c["issues"] else "")}


@traced("confidence_gate")
def confidence_gate(state):
    _, backend = get_embedding()
    ref = 0.35 if backend == "simple" else 0.60      # typical "good match" similarity per backend
    retrieval = min(1.0, state.get("top_sim", 0.0) / ref)
    crit = state.get("critique", {})
    conf = round(0.4 * retrieval + 0.6 * float(crit.get("score", 0.0)), 2)
    ok = crit.get("grounded") and conf >= config.CONFIDENCE_THRESHOLD and state["draft"] != "NOT_IN_NOTES"
    return {"confidence": conf,
            "_info": f"confidence {conf:.2f} (threshold {config.CONFIDENCE_THRESHOLD}) -> "
                     + ("answer" if ok else "ESCALATE")}


@traced("output_guard")
def output_guard(state):
    text, events, ok = guardrails.check_output(state["draft"], state["mode"], bool(state.get("chunks")))
    out = {"guard_events": events, "output_ok": ok,
           "_info": "passed" if ok else "WITHHELD: " + events[-1]["guardrail"]}
    if ok:
        if state.get("scope_note"):
            text = f"_{state['scope_note']}_\n\n{text}"
        out.update(final_answer=text, status="hint" if state["mode"] == "hint" else "answered",
                   answered_by="agent")
    return out


def _record(state, **extra):
    """Common fields when writing this doubt to SQLite."""
    draft = state.get("draft", "")
    rec = {"id": state["doubt_id"], "student": state["student"], "question": state.get("question") or "",
           "category": state.get("category", ""), "topic": state.get("topic", ""),
           "draft": "" if draft == "NOT_IN_NOTES" else draft, "sources": _sources(state),
           "confidence": state.get("confidence", 0.0), "guardrails": state.get("guard_events", []),
           "created_at": state["created_at"], "standalone": _q(state),
           "chat_id": state.get("chat_id"), "scope": state.get("scope", ""), "subject": _subject(state)}
    rec.update(extra)
    return rec


@traced("confirm_student")
def confirm_student(state):
    # HUMAN-IN-THE-LOOP #1: the STUDENT approves before anything is sent to a TA.
    store.save(_record(state, status="awaiting_student", answer="", answered_by="", resolved_at=None,
                       trace=state.get("trace", []) + [{"node": "confirm_student",
                                                        "info": "PAUSED: asking student", "ms": 0}]))
    reply = interrupt({"type": "student_confirm", "doubt_id": state["doubt_id"]})
    if reply.get("send"):
        return {"student_note": reply.get("note", ""), "_info": "student approved sending to TA"}
    return {"status": "withdrawn", "mode": "withdrawn", "answered_by": "student",
            "final_answer": "Not sent to a TA (you chose to keep this doubt to yourself).",
            "_info": "student cancelled"}


@traced("notify_ta")
def notify_ta(state):
    """Write the doubt to the TA queue before pausing."""
    note = state.get("student_note", "").strip()
    store.save(_record(state, status="escalated", answer="", answered_by="", resolved_at=None,
                       draft=(f"Student's note: {note}\n\n" if note else "") +
                             ("" if state.get("draft", "") == "NOT_IN_NOTES" else state.get("draft", "")),
                       trace=state.get("trace", []) + [{"node": "notify_ta", "info": "added to TA queue", "ms": 0},
                                                       {"node": "wait_for_ta", "info": "PAUSED (interrupt)", "ms": 0}]))
    return {"status": "escalated", "mode": "ta", "_info": "added to TA queue"}


@traced("wait_for_ta")
def wait_for_ta(state):
    # HUMAN-IN-THE-LOOP: execution stops here and is checkpointed to SQLite.
    # It resumes when the TA submits an answer: graph.invoke(Command(resume={...}))
    reply = interrupt({"doubt_id": state["doubt_id"], "question": state["question"],
                       "draft": state.get("draft", ""), "confidence": state.get("confidence", 0.0)})
    return {"ta_answer": reply["answer"], "ta_name": reply.get("ta", "TA"),
            "ta_save": reply.get("save", True),
            "_info": f"resumed by {reply.get('ta', 'TA')}"}


@traced("learn")
def learn(state):
    text, events = guardrails.mask_pii(state["ta_answer"])
    if state.get("ta_save", True):
        knowledge.add_ta_answer(_q(state), text, state["ta_name"], state.get("topic", ""),
                                state["doubt_id"], subject=_subject(state))
        info = "TA answer stored in vector memory"
    else:
        info = "TA replied to this student only (not saved to memory)"
    return {"final_answer": text, "status": "ta_answered", "answered_by": f"ta:{state['ta_name']}",
            "guard_events": events, "_info": info}


@traced("finalize")
def finalize(state):
    store.save(_record(state, status=state.get("status", ""), answer=state.get("final_answer", ""),
                       draft=state.get("draft", ""), answered_by=state.get("answered_by", ""),
                       resolved_at=store.now(),
                       trace=state.get("trace", []) + [{"node": "finalize", "info": state.get("status", ""),
                                                        "ms": 0}]))
    return {"_info": state.get("status", "")}


def _sources(state):
    if state.get("mode") == "memory" and state.get("memory_hit"):
        m = state["memory_hit"]["meta"]
        return [{"label": f"Verified by {m['ta']} on {m['saved_at']}", "sim": state["memory_hit"]["sim"]}]
    out = []
    for i, c in enumerate(state.get("chunks", []), start=1):
        m = c["meta"]
        title = knowledge.scope_label(m["source"])
        where = f"page {m['page']}" if m.get("page") else m["section"]
        out.append({"label": f"S{i} · {title} · {where}", "sim": c["sim"], "text": c["text"][:600],
                    "file": m["source"], "page": m.get("page") or 0, "unit": m["unit"], "title": title,
                    "section": m["section"]})
    return out


# ------------------------------------------------------------------ edges
def after_input(state):
    return "finalize" if state.get("block_reason") else "contextualize"


def after_context(state):
    return "finalize" if state.get("mode") == "history" else "retrieve"


def after_router(state):
    return {"off_topic": "refuse", "smalltalk": "refuse", "library": "refuse", "admin": "confirm_student",
            "assignment": "answer_agent", "summary": "summarize"}.get(state["category"], "ta_memory")


def after_summary(state):
    return "confirm_student" if state.get("draft") == "NOT_IN_NOTES" else "output_guard"


def after_memory(state):
    return "finalize" if state.get("memory_hit") else "answer_agent"


def after_answer(state):
    return "output_guard" if state.get("mode") == "hint" else "critic_agent"


def after_critic(state):
    c = state["critique"]
    if not c["grounded"] and state["draft"] != "NOT_IN_NOTES" and state["retries"] <= config.MAX_CRITIC_RETRIES:
        return "answer_agent"                      # reflection: rewrite once
    return "confidence_gate"


def after_gate(state):
    crit = state.get("critique", {})
    ok = crit.get("grounded") and state.get("confidence", 0) >= config.CONFIDENCE_THRESHOLD \
        and state.get("draft") != "NOT_IN_NOTES"
    return "output_guard" if ok else "confirm_student"


def after_output(state):
    return "finalize" if state.get("output_ok") else "confirm_student"


def after_confirm(state):
    return "finalize" if state.get("status") == "withdrawn" else "notify_ta"


def build_graph():
    g = StateGraph(DoubtState)
    for name, fn in [("input_guard", input_guard), ("contextualize", contextualize), ("retrieve", retrieve), ("router", router),
                     ("refuse", refuse), ("summarize", summarize), ("ta_memory", ta_memory), ("answer_agent", answer_agent),
                     ("critic_agent", critic_agent), ("confidence_gate", confidence_gate),
                     ("output_guard", output_guard), ("confirm_student", confirm_student), ("notify_ta", notify_ta),
                     ("wait_for_ta", wait_for_ta), ("learn", learn), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", after_input, ["contextualize", "finalize"])
    g.add_conditional_edges("contextualize", after_context, ["retrieve", "finalize"])
    g.add_edge("retrieve", "router")
    g.add_conditional_edges("router", after_router,
                            ["refuse", "confirm_student", "answer_agent", "ta_memory", "summarize"])
    g.add_conditional_edges("summarize", after_summary, ["output_guard", "confirm_student"])
    g.add_edge("refuse", "finalize")
    g.add_conditional_edges("ta_memory", after_memory, ["finalize", "answer_agent"])
    g.add_conditional_edges("answer_agent", after_answer, ["output_guard", "critic_agent"])
    g.add_conditional_edges("critic_agent", after_critic, ["answer_agent", "confidence_gate"])
    g.add_conditional_edges("confidence_gate", after_gate, ["output_guard", "confirm_student"])
    g.add_conditional_edges("output_guard", after_output, ["finalize", "confirm_student"])
    g.add_conditional_edges("confirm_student", after_confirm, ["notify_ta", "finalize"])
    g.add_edge("notify_ta", "wait_for_ta")
    g.add_edge("wait_for_ta", "learn")
    g.add_edge("learn", "finalize")
    g.add_edge("finalize", END)
    return g


@lru_cache(maxsize=1)
def get_app():
    conn = sqlite3.connect(config.CHECKPOINT_PATH, check_same_thread=False)
    return build_graph().compile(checkpointer=SqliteSaver(conn))


def graph_dot() -> str:
    """Graphviz DOT of the compiled graph (rendered in the Faculty view and README)."""
    g = get_app().get_graph()
    lines = ['digraph G { rankdir=TB; bgcolor="transparent"; '
             'node [shape=box, style="rounded,filled", fillcolor="#1E1F52", color="#B69CFF", '
             'fontcolor="#ECEAFF", fontname="Helvetica", fontsize=11, penwidth=1.2]; '
             'edge [color="#7E7BB8", arrowsize=0.7];']
    for nid in g.nodes:
        if nid in ("__start__", "__end__"):
            lines.append(f'"{nid}" [label="{nid.strip("_")}", shape=ellipse, color="#6EE7D8", fontcolor="#6EE7D8"];')
        elif nid in ("wait_for_ta", "notify_ta", "learn", "confirm_student"):
            lines.append(f'"{nid}" [color="#FFD18A", fontcolor="#FFD18A"];')
        elif nid in ("input_guard", "output_guard", "refuse"):
            lines.append(f'"{nid}" [color="#FF9EC7", fontcolor="#FF9EC7"];')
    for e in g.edges:
        style = ' [style=dashed]' if e.conditional else ""
        lines.append(f'"{e.source}" -> "{e.target}"{style};')
    lines.append("}")
    return "\n".join(lines)