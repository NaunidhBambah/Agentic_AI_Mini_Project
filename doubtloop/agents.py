"""The worker agents: Router, Answer, Critic, Summary, Librarian, Organiser and Insight agents.

Every agent has an LLM version and a simple offline fallback, so a missing key or a
rate limit never crashes the demo (the fallback is recorded in the trace).
"""
import re

from . import config, llm

TOPICS = {
    "Agents vs chatbots": ["agent", "chatbot", "assistant", "autonomy", "autonomous"],
    "ReAct & reasoning loops": ["react", "thought", "observation", "reason", "loop"],
    "Function calling & tools": ["function", "tool", "schema", "json", "calling", "mcp", "protocol"],
    "Agent memory": ["memory", "episodic", "semantic", "context window", "long-term", "short-term"],
    "Planning": ["plan", "planning", "decomposition", "plan-and-execute", "subtask"],
    "LangChain & LlamaIndex": ["langchain", "llamaindex", "chain", "lcel", "loader", "retriever"],
    "Vector databases": ["faiss", "chroma", "chromadb", "pinecone", "vector", "index"],
    "RAG & chunking": ["rag", "retrieval", "chunk", "chunking", "ingestion", "augmented"],
    "Embeddings & similarity": ["embedding", "cosine", "similarity", "dense", "vectorize"],
    "Multi-agent systems": ["multi-agent", "orchestrator", "worker", "autogen", "crewai", "crew", "langgraph"],
    "Reflection & self-critique": ["reflection", "critique", "critic", "self-correct"],
    "Human-in-the-loop": ["human", "hitl", "approval", "interrupt", "feedback"],
    "Evaluation & observability": ["evaluation", "metric", "trajectory", "langsmith", "langfuse", "observability", "trace"],
    "Guardrails & safety": ["guardrail", "nemo", "safety", "hallucination", "injection", "pii"],
    "Ethics & privacy": ["ethics", "gdpr", "privacy", "bias", "model card"],
    "Deployment": ["deploy", "fastapi", "docker", "streamlit", "hugging face", "render", "railway", "react frontend"],
}
ADMIN_WORDS = re.compile(r"\b(exam date|deadline|due date|marks|attendance|submission|submit|"
                         r"timetable|syllabus copy|re-?exam|grade sheet|when is|extension)\b", re.I)
COURSE_WORDS = {w for kws in TOPICS.values() for w in kws} | {
    "llm", "prompt", "model", "agentic", "ai", "token", "context", "gpt", "langgraph", "crew"}


# ------------------------------------------------------------------ router
def _keyword_topic(q: str, extra=()) -> str:
    """Best topic for a question: built-in AI topics plus the topics the librarian found in the uploads."""
    ql = q.lower()
    best, score = "General", 0
    for topic, kws in TOPICS.items():
        s = sum(1 for k in kws if k in ql)
        if s > score:
            best, score = topic, s
    words = set(re.findall(r"[a-z0-9]{3,}", ql))
    for topic in extra:
        tw = {w for w in re.findall(r"[a-z0-9]{3,}", topic.lower()) if w not in {"and", "the", "for"}}
        s = len(tw & words) + (1 if topic.lower() in ql else 0)
        if tw and s >= max(1, score + (0 if best == "General" else 1)) and s > score:
            best, score = topic, s
    return best


def _has_course_words(question: str) -> bool:
    words = re.findall(r"[a-z\-]+", question.lower())
    return any(w.startswith(c) for w in words for c in COURSE_WORDS if len(c) > 2)


SUMMARY_Q = re.compile(r"\b(summari[sz]e|summary|overview|key points|main points|bullet points|outline|"
                       r"recap|short notes|revision notes|gist|(main|important|key|major) topics|"
                       r"topics (to study|covered|in (this|the) (unit|file|document|notes|chapter))|"
                       r"what (should|do) i (study|prepare)|what to (study|prepare)|cheat ?sheet)\b", re.I)
MATERIAL_Q = re.compile(r"\b(notes?|lectures?|slides?|documents?|pdfs?|files?|units? ?\d?|chapters?|modules?|"
                        r"syllabus|lab manual|handout|material|page \d+)\b", re.I)
SMALLTALK = re.compile(r"^\s*(hi+|hello+|hey+|hii+|good (morning|afternoon|evening)|thanks?( you)?|thank u|"
                       r"ok(ay)?|bye|namaste)\W*$", re.I)
# Chit-chat inside a longer sentence ("Hi, how are you?", "Can you greet me good morning first?")
CHITCHAT = re.compile(r"^\s*(hi+|hello+|hey+|hii+|namaste|good (morning|afternoon|evening|night))\b|"
                      r"\bhow are you\b|\bgreet me\b|\bwho are you\b|\bwhat can you do\b|\bnice to meet\b|"
                      r"\bwhat'?s up\b|\bthank(s| you)\b", re.I)


def is_smalltalk(question: str, top_sim: float = 0.0, floor: float = 0.3) -> bool:
    """A greeting or chit-chat with no real course question inside it."""
    if SMALLTALK.search(question):
        return True
    if not CHITCHAT.search(question) or len(question.split()) > 18:
        return False
    rest = CHITCHAT.sub(" ", question)
    has_question = bool(re.search(r"\b(what|why|how|explain|define|difference|list|summari[sz]e)\b", rest, re.I))
    return not has_question and top_sim < floor + 0.15 and not _has_course_words(rest)


# Questions about the material itself ("can you see the selected notes?", "which files are uploaded?")
LIBRARY_Q = re.compile(
    r"\b(can|do) you (see|read|access|have)\b.*\b(notes?|files?|documents?|pdfs?|materials?|slides?)\b|"
    r"\b(what|which) (notes?|files?|documents?|pdfs?|materials?|units?) (do you have|are (there|uploaded|available|"
    r"selected)|have been uploaded|can you see|is (this|selected))\b|"
    r"\bwhat (is|are) (in )?(the |this |these )?(selected|uploaded|available) (notes?|files?|documents?|materials?)\b",
    re.I)


def is_summary_request(question: str) -> bool:
    return bool(SUMMARY_Q.search(question))


def route(question: str, top_sim: float, assignment_hit: bool, retrieval_floor: float,
          sections=None, scoped: bool = False, library: str = "", lib_topics=()) -> dict:
    """Classify the doubt -> {category, topic, reason, by}.

    Rule-first: clear cases are decided instantly without an LLM call (faster);
    only ambiguous questions go to the LLM router.
    """
    topic = _keyword_topic(question, lib_topics)
    allowed = list(dict.fromkeys([*lib_topics, *TOPICS]))
    if len(re.findall(r"[A-Za-z]", question)) < 4:          # "Ma", "?", "ok" sent by mistake
        return {"category": "smalltalk", "topic": "Greeting", "reason": "too short to be a question", "by": "rule"}
    if assignment_hit:
        return {"category": "assignment", "topic": topic, "reason": "matches a graded assignment question", "by": "rule"}
    if LIBRARY_Q.search(question):
        return {"category": "library", "topic": "Library", "reason": "asks what material is available", "by": "rule"}
    if is_smalltalk(question, top_sim, retrieval_floor):
        return {"category": "smalltalk", "topic": "Greeting", "reason": "greeting / chit-chat", "by": "rule"}
    if is_summary_request(question) and (scoped or MATERIAL_Q.search(question) or top_sim >= retrieval_floor):
        return {"category": "summary", "topic": topic, "reason": "asks to summarise course material", "by": "rule"}
    strong = top_sim >= retrieval_floor + 0.2
    if ADMIN_WORDS.search(question) and not strong:
        return {"category": "admin", "topic": "Admin", "reason": "admin keywords", "by": "rule"}
    if scoped or strong or MATERIAL_Q.search(question):
        why = "student focused on a document" if scoped else ("close match in the notes" if strong
                                                                else "refers to course material")
        return {"category": "concept", "topic": topic, "reason": why, "by": "rule"}
    if llm.is_online():
        try:
            nearest = ", ".join(sections or []) or "none"
            out = llm.chat_json(
                "You route student messages for a university course helpdesk.\n"
                + (f"The uploaded course materials cover:\n{library}\n" if library else "")
                + "Read every term in the course's sense (for an AI course: 'memory' = agent memory). "
                "If in doubt between concept and off_topic, choose concept.\n"
                "Categories:\n"
                "- concept: a doubt about anything the course or its materials cover\n"
                "- assignment: asks for a full solution/code to a lab, assignment or exam question\n"
                "- admin: deadlines, marks, attendance, exam dates, submissions\n"
                "- off_topic: clearly unrelated to the course (movies, cricket, recipes, chit-chat)\n"
                f"Pick topic from this list or 'General': {allowed[:40]}\n"
                'JSON keys: "category", "topic", "reason" (max 12 words).',
                f"Student message: {question}\nClosest course-note sections: {nearest}")
            cat = str(out.get("category", "concept")).lower().strip()
            if cat not in {"concept", "assignment", "admin", "off_topic"}:
                cat = "concept"
            topic = out.get("topic") if out.get("topic") in allowed else topic
            reason = out.get("reason", "")
            # Evidence override: never refuse a question that clearly matches the course notes.
            if cat == "off_topic" and (top_sim >= retrieval_floor + 0.15 or _has_course_words(question)):
                cat, reason = "concept", "LLM said off-topic, overridden: matches course notes"
            return {"category": cat, "topic": topic, "reason": reason, "by": "llm"}
        except llm.LLMError:
            pass
    # offline rules
    tech_term = re.search(r"\b[A-Za-z0-9]*([A-Z][A-Za-z0-9]*[A-Z]|[a-z][A-Z]|\d)[A-Za-z0-9]*\b", question)
    # No course words and no technical term -> off-topic. Unknown tech terms go to a human instead.
    if not tech_term and not _has_course_words(question) and top_sim < retrieval_floor + 0.15:
        return {"category": "off_topic", "topic": "Off-topic", "reason": "no course terms", "by": "rule"}
    return {"category": "concept", "topic": topic, "reason": "course terms found", "by": "rule"}


# ------------------------------------------------------------------ summary agent
SUMMARY_SYSTEM = (
    "You are DoubtLoop, a teaching assistant. Summarise ONLY the numbered course-note sources. "
    "Start directly (no greeting, no 'Here is a summary'). Group points under bold headings in the order "
    "of the notes, follow the student's requested format (if they ask for N points give EXACTLY N "
    "bullets; if they ask for the main topics, list the topics with one line each), and cite every "
    "bullet with [S1] or [S1][S2] (never [S1, S2]). Write formulas in LaTeX ($...$). "
    "Never add facts that are not in the sources. Max 300 words.")


_MULTI_CITE = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S?\d+)+\s*\]")


def norm_citations(text: str) -> str:
    """'[S3, S4]' -> '[S3][S4]' so every citation can be checked and shown as a clickable chip."""
    def split(m):
        nums = re.findall(r"\d+", m.group(0))
        return "".join(f"[S{n}]" for n in nums)
    return _MULTI_CITE.sub(split, text)


def summarize(question: str, chunks, conversation: str = "", on_token=None) -> dict:
    if not chunks:
        return {"text": "NOT_IN_NOTES", "by": "rule"}
    user = f"Sources:\n{format_context(chunks)}\n\n"
    if conversation:
        user += f"Earlier in this chat:\n{conversation}\n\n"
    user += f"Student request: {question}"
    if llm.is_online():
        try:
            return {"text": clean_answer(llm.chat(SUMMARY_SYSTEM, user, long=True, on_token=on_token)), "by": "llm"}
        except llm.LLMError as exc:
            fallback = f"fallback ({str(exc)[:60]})"
        else:
            fallback = "offline"
    else:
        fallback = "offline"
    bullets = []
    for i, c in enumerate(chunks[:10], start=1):
        body = c["text"].split("\n", 1)[-1] if c["text"].startswith(c["meta"].get("section", "\0")) else c["text"]
        sent = next((x.strip(" -*#") for x in re.split(r"(?<=[.!?])\s+", body.replace("\n", " "))
                     if len(x.strip()) > 40 and "(contd.)" not in x), "")
        if sent:
            bullets.append(f"- {sent} [S{i}]")
    return {"text": "**Summary of the notes (offline mode)**\n\n" + "\n".join(bullets), "by": fallback}


# ------------------------------------------------------------------ librarian agent
UNIT_GUIDE = {
    "1": "Foundations: agents vs chatbots, ReAct, function calling, tool schemas, memory, planning",
    "2": "LLM frameworks and RAG: LangChain, LlamaIndex, FAISS/Chroma/Pinecone, chunking, embeddings",
    "3": "Multi-agent systems: orchestrator-worker, AutoGen, CrewAI, reflection, human-in-the-loop",
    "4": "Evaluation and safety: metrics, guardrails, NeMo, LangSmith/LangFuse, failure modes, ethics",
    "5": "Deployment: FastAPI, React, Docker, Hugging Face Spaces, Render, integration",
}
TOPIC_UNIT = {"Agents vs chatbots": "1", "ReAct & reasoning loops": "1", "Function calling & tools": "1",
              "Agent memory": "1", "Planning": "1", "LangChain & LlamaIndex": "2", "Vector databases": "2",
              "RAG & chunking": "2", "Embeddings & similarity": "2", "Multi-agent systems": "3",
              "Reflection & self-critique": "3", "Human-in-the-loop": "3", "Evaluation & observability": "4",
              "Guardrails & safety": "4", "Ethics & privacy": "4", "Deployment": "5"}
KINDS = ["Lecture notes", "Slides", "Lab manual", "Assignment", "Question paper", "Syllabus", "Reference",
         "Other"]


def _sample(text: str, n: int = 4500) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    if len(text) <= n:
        return text
    return text[: int(n * 0.7)] + "\n...\n" + text[len(text) // 2: len(text) // 2 + int(n * 0.3)]


def _offline_catalog(filename: str, text: str) -> dict:
    head = text[:1500]
    lines = [ln.strip() for ln in text.splitlines()]
    repeated = {ln for ln in set(lines) if ln and lines.count(ln) >= 3}     # slide header on every page
    title = next((ln.strip(" #:-") for ln in lines
                  if ln not in repeated and 3 <= len(ln.split()) <= 14 and len(ln) < 100), "") or \
        re.sub(r"[_-]+", " ", filename.rsplit(".", 1)[0]).strip().title()
    low = (filename + " " + head).lower()
    kind = ("Lab manual" if re.search(r"\b(lab|practical|experiment)\b", low) else
            "Assignment" if re.search(r"\b(assignment|homework)\b", low) else
            "Question paper" if re.search(r"\b(question paper|end.?sem|mid.?sem|marks\))\b", low) else
            "Slides" if "slide" in low else "Lecture notes")
    tl = text.lower()
    scores = sorted(((sum(tl.count(k) for k in kws), t) for t, kws in TOPICS.items()), reverse=True)
    topics = [t for n, t in scores[:5] if n > 0]
    unit = _explicit_unit(filename, text) or (TOPIC_UNIT.get(topics[0], "") if topics else "")
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(x.strip()) > 40]
    return {"title": title[:90], "unit": unit, "kind": kind, "topics": topics,
            "summary": " ".join(sents[:2])[:300], "by": "rules"}


def _explicit_unit(filename: str, text: str) -> str:
    """'Unit-2_notes.pdf' or a first page saying 'Unit 2' -> '2' (works for any subject)."""
    m = re.search(r"unit[\s_-]*0?([1-9])\b", filename, re.I) or re.search(r"\bunit[\s_-]*0?([1-9])\b", text[:800], re.I)
    return m.group(1) if m else ""


def catalog_document(filename: str, text: str, pages: int, subject: str = None) -> dict:
    """Librarian agent: read an uploaded file and file it (title, unit, type, topics, summary)."""
    base = _offline_catalog(filename, text)
    if not llm.is_online():
        return base
    subject = subject or config.COURSE_NAME
    agentic = "agentic" in subject.lower()              # the built-in unit guide is for the Agentic AI course
    try:
        out = llm.chat_json(
            f"You are the librarian agent for the university subject '{subject}'. Read the start of an "
            "uploaded file and catalogue it.\n"
            + ("Course units:\n" + "\n".join(f"Unit {k}: {v}" for k, v in UNIT_GUIDE.items()) + "\n"
               if agentic else "")
            + "unit: the unit/module number if the file says it (e.g. 'Unit-2', 'Module 3')"
            + (", otherwise the best matching unit above" if agentic else "")
            + "; \"\" if it is not tied to one unit (syllabus, reference book, question bank).\n"
            f"kind: one of {KINDS}.\n"
            "Describe only what the file contains. Never comment on its file name or on whether it fits "
            "the subject.\n"
            'JSON keys: "title" (short, human), "unit", "kind", "topics" (3-6 short topics), '
            '"summary" (2 sentences).',
            f"File name: {filename}\nPages: {pages}\n\nContent:\n{_sample(text)}")
        unit = re.sub(r"(?i)unit|module", "", str(out.get("unit", ""))).strip()
        explicit = _explicit_unit(filename, text)
        return {"title": str(out.get("title") or base["title"])[:90],
                "unit": explicit or (unit if unit.isdigit() and len(unit) <= 2 else ""),
                "kind": out.get("kind") if out.get("kind") in KINDS else base["kind"],
                "topics": [str(t)[:40] for t in (out.get("topics") or base["topics"])][:6],
                "summary": str(out.get("summary") or base["summary"])[:400], "by": "llm"}
    except llm.LLMError:
        return base


def organize_library(docs: list, instruction: str) -> dict:
    """Organiser agent: propose a new arrangement of the library from the teacher's instruction.

    Returns {"changes": [{file, title, unit, kind, collection, reason}], "duplicates": [[a, b]], "note": str}.
    Nothing is applied until the teacher approves (human-in-the-loop).
    """
    import difflib

    files = {d["file"]: d for d in docs}
    listing = "\n".join(
        f"- file={d['file']} | title={d['title']} | unit={d['unit']} | type={d['kind']} | "
        f"collection={d['collection'] or '-'} | topics={', '.join(d['topics'][:5])} | summary={d['summary'][:160]}"
        for d in docs)
    plan = None
    if llm.is_online() and docs:
        try:
            out = llm.chat_json(
                "You are the organiser agent for a course library. Follow the teacher's instruction to arrange "
                "the documents. You may change title, unit, type and collection (a folder name). Keep changes "
                "minimal and explain each in under 12 words. Course units:\n"
                + "\n".join(f"Unit {k}: {v}" for k, v in UNIT_GUIDE.items()) + f"\nAllowed types: {KINDS}.\n"
                'JSON keys: "changes" (list of {"file","title","unit","kind","collection","reason"}, only for '
                'documents that change), "duplicates" (list of [file, file] pairs that look like the same '
                'material), "note" (one sentence summary of the arrangement).',
                f"Teacher's instruction: {instruction or 'Arrange the library neatly by unit and type.'}\n\n"
                f"Documents:\n{listing}", fast=False)
            plan = {"changes": [], "duplicates": [], "note": str(out.get("note", "")), "by": "llm"}
            for c in out.get("changes") or []:
                f = c.get("file")
                if f not in files:
                    continue
                unit = str(c.get("unit", "")).replace("Unit", "").strip()
                plan["changes"].append({
                    "file": f, "title": str(c.get("title") or files[f]["title"])[:90],
                    "unit": unit if unit in UNIT_GUIDE else "",
                    "kind": c.get("kind") if c.get("kind") in KINDS else files[f]["kind"],
                    "collection": str(c.get("collection") or "")[:40], "reason": str(c.get("reason", ""))[:100]})
            plan["duplicates"] = [p for p in (out.get("duplicates") or [])
                                  if isinstance(p, list) and len(p) == 2 and all(x in files for x in p)]
        except llm.LLMError:
            plan = None
    if plan is None:                                   # offline: folder per type, flag near-identical titles
        plan = {"changes": [], "duplicates": [], "by": "rules",
                "note": "Grouped each document into a collection by its type (offline mode)."}
        for d in docs:
            coll = {"Lecture notes": "Lectures", "Slides": "Lectures", "Lab manual": "Labs",
                    "Assignment": "Assignments", "Question paper": "Past papers"}.get(d["kind"], "Reference")
            if d["collection"] != coll:
                plan["changes"].append({"file": d["file"], "title": d["title"],
                                        "unit": re.sub(r"\D", "", d["unit"]), "kind": d["kind"],
                                        "collection": coll, "reason": f"{d['kind']} go in {coll}"})
        names = list(files)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if difflib.SequenceMatcher(None, files[a]["title"].lower(), files[b]["title"].lower()).ratio() > 0.85:
                    plan["duplicates"].append([a, b])
    return plan


# ------------------------------------------------------------------ conversation memory
HISTORY_Q = re.compile(
    r"\b(what|which|show|list|tell)\b.*\b(my|i)\b.*\b(first|last|previous|earlier|past|all|asked)\b.*\b(question|doubt|ask|asked)|"
    r"\bwhat (did|have) i (ask|asked)\b|\bmy (previous|earlier|last|first|past) (question|doubt)s?\b", re.I)
FOLLOW_UP = re.compile(r"\b(it|this|that|these|those|them|they|its|their|above|same|previous)\b|"
                       r"^\s*(and|so|ok|okay|also|then|what about|how about)\b", re.I)
STARTER = re.compile(r"^\s*(and|so|ok|okay|also|then|what about|how about)\b", re.I)


def is_history_question(question: str) -> bool:
    return bool(HISTORY_Q.search(question))


def format_turns(turns: list, max_turns: int = 3) -> str:
    """Last few Q/A pairs of THIS chat, compact, for conversation memory."""
    out = []
    for t in turns[-max_turns:]:
        a = re.sub(r"\s+", " ", (t.get("answer") or "(no answer yet)"))[:280]
        out.append(f"Student: {t.get('standalone') or t['question']}\nDoubtLoop: {a}")
    return "\n\n".join(out)


def contextualize(question: str, turns: list) -> dict:
    """Rewrite a follow-up ("how much should I study it?") into a standalone question
    using the earlier questions AND answers of this chat (conversation memory)."""
    probe = re.sub(r"\b(this|these|that|those|the)\s+(selected\s+)?(notes?|lectures?|slides?|documents?|files?|"
                   r"pdfs?|units?|chapters?|modules?|materials?|topics?)\b", " ", question, flags=re.I)
    if not turns or not FOLLOW_UP.search(probe):
        return {"standalone": question, "by": "unchanged"}
    if llm.is_online():
        try:
            out = llm.chat_json(
                "Rewrite the student's latest message into ONE standalone question that makes sense "
                "without the conversation, by replacing pronouns like 'it/this/them/that' with what they "
                "refer to in the conversation. Keep every instruction about format or length (e.g. 'in "
                "detail', 'a paragraph for each step', 'with examples', 'in a table') word for word. "
                "If it is already standalone, return it unchanged. "
                'JSON key: "standalone".',
                f"Conversation so far:\n{format_turns(turns)}\n\nLatest message: {question}")
            text = str(out.get("standalone", "")).strip()
            if text:
                return {"standalone": text, "by": "llm"}
        except llm.LLMError:
            pass
    # Offline fallback: only clear follow-ups ("okay, and its limits?"), never long self-contained questions
    last = turns[-1].get("standalone") or turns[-1]["question"]
    if STARTER.search(question) or len(question.split()) <= 6:
        return {"standalone": f"{question} (context: {last})", "by": "rule"}
    return {"standalone": question, "by": "unchanged"}


def history_answer(past: list) -> str:
    """Answer 'what was my first question?' from this chat's history (episodic memory)."""
    if not past:
        return "This is the first doubt in this chat."
    lines = [f"{i}. {d['question']}  _({d['created_at'][5:16]}, {d['status'].replace('_', ' ')})_"
             for i, d in enumerate(past, start=1)]
    return (f"Your first doubt in this chat was: **{past[0]['question']}**\n\nAll doubts in this chat:\n\n"
            + "\n".join(lines[:15]))


# ------------------------------------------------------------------ answer agent
def format_context(chunks) -> str:
    return "\n\n".join(
        f"[S{i}] ({c['meta']['unit']} · {c['meta']['section']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1))


STYLES = {   # answer length the student picks (clearly different, so the choice is visible)
    "brief": "LENGTH: Brief, 50-100 words. Only the key points as short bullets. No examples, no headings.",
    "detailed": ("LENGTH: Detailed, 250-450 words. For every key point write 2-3 full teaching sentences: "
                 "what it is, why it matters, how it works, with the notes' own examples."),
    "exam": ("LENGTH: Exam answer, 500-800 words, written like a top-scoring university exam answer with "
             "bold headings: Introduction/Definition, Explanation (a paragraph for each point), Steps / Types "
             "/ Components, Example, Advantages and Limitations, Conclusion. Use the headings the notes have "
             "material for."),
}
_LONG_ASK = re.compile(r"\b(in (more |full |great )?detail|detailed|elaborate|paragraphs?|long answer|explain each|"
                       r"each step in|step by step|exam answer|\d+ marks?)\b", re.I)
_SHORT_ASK = re.compile(r"\b(briefly|in brief|short answer|in short|one line|one sentence|two lines|quick|"
                        r"tl;?dr|in a nutshell)\b", re.I)


def effective_style(question: str, style: str) -> str:
    """The student's own words win over the length buttons ('explain in detail' -> longer)."""
    if _SHORT_ASK.search(question):
        return "brief"
    if _LONG_ASK.search(question):
        return "exam" if style == "exam" or re.search(r"\b(exam|\d+ marks?|paragraphs?)\b", question, re.I) \
            else ("exam" if style == "detailed" else "detailed")
    return style


def answer_system(subject: str = None, style: str = "detailed") -> str:
    subject = subject or config.COURSE_NAME
    return (
        f"You are DoubtLoop, an expert teaching assistant for the university subject {subject}. "
        "A student asked a doubt; the numbered sources [S1], [S2]... are excerpts from the course notes.\n"
        "HOW TO ANSWER\n"
        "- Start directly with the answer. No greeting, no 'Hello', no 'As your teaching assistant', no "
        "'Based on the provided sources', no closing offers like 'Let me know'.\n"
        "- Use ALL relevant sources together: combine points spread over several slides into one complete, "
        "well-ordered explanation (e.g. list every step, every principle, every type the notes give).\n"
        "- Expand the notes' short bullet points into clear teaching sentences, but every fact must come "
        "from the sources. Do not add outside facts, numbers or examples.\n"
        "- Use plain, simple English a second-year student understands. Never pad: if the notes say only a "
        "line or two about the topic, give a short clear answer instead of repeating the same point in "
        "grander words.\n"
        "- Keep the notes' own labels and numbering (if the notes say 'Step 4', call it Step 4).\n"
        "- Format for learning: a one-line definition first, then numbered steps for processes, a markdown "
        "table for comparisons, bullets for lists, and formulas in LaTeX ($EV = PV \\times \\%complete$ "
        "inline or $$...$$ on its own line) with each symbol explained. Bold the key terms.\n"
        "- Cite after each point as [S1] or [S1][S2] (never [S1, S2]).\n"
        "- FOLLOW THE STUDENT'S OWN INSTRUCTIONS about format first: a paragraph per step, a table, N "
        "points, simpler words, more detail. They override the default length and layout below.\n"
        "- If the student asks for examples and the notes have none for a point, you may add ONE short, "
        "simple example on its own line starting with '*Example (added for clarity, not from your notes):*'. "
        "Never present such an example as coming from the notes.\n"
        "- If the notes cover only part of the question, answer that part fully and end with one line: "
        "'*Not covered in your notes:* <the missing part>'. Never print NOT_IN_NOTES in that case.\n"
        "- Reply with exactly NOT_IN_NOTES (nothing else) only when the sources contain nothing that answers "
        "the question.\n"
        + STYLES.get(style, STYLES["detailed"]))


ANSWER_SYSTEM = answer_system()       # kept for older imports

HINT_SYSTEM = (
    "You are DoubtLoop. The student asked about a GRADED assignment/lab. Academic-integrity rule: "
    "never give the full solution or complete code. Give 2-4 guiding hints, point to the relevant "
    "concept in the sources with [S1]-style citations, and suggest what to try first. Max 120 words.")


_COMMON = set("""what why how when which who does difference between explain related relation use
used using work works working mean means meaning example give tell about need same this that with
from into have there their them they your than then also more most best better good just like
can could should would will make made other some many much very really actually please""".split())


def _offline_answer(question: str, chunks, hint: bool) -> str:
    from .knowledge import notes_vocab

    # Honesty check: a technical term (acronym, CamelCase, or with digits) that the notes never
    # mention means the notes can't answer the question, so escalate instead of guessing.
    clean = re.sub(r"\[[A-Z_ ]+MASKED\]", " ", question)
    terms = {t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", clean)
             if re.search(r"[A-Z].*[A-Z]|[a-z][A-Z]|\d", t)}
    vocab = notes_vocab()
    unknown = {t for t in terms
               if any(p not in vocab and p.rstrip("s") not in vocab for p in t.lower().split("-") if p)}
    if not hint and unknown:
        return "NOT_IN_NOTES"
    qwords = {w for w in re.findall(r"[a-z]{4,}", question.lower())}
    scored = []
    for i, c in enumerate(chunks[:3], start=1):
        for sent in re.split(r"(?<=[.!?])\s+|\n", c["text"]):
            sent = sent.strip(" -*#")
            if len(sent) < 30:
                continue
            overlap = len(qwords & set(re.findall(r"[a-z]{4,}", sent.lower())))
            scored.append((overlap, i, sent))
    scored.sort(key=lambda x: -x[0])
    picked = [s for s in scored if s[0] > 0][:3] or scored[:2]
    if not picked:
        return "NOT_IN_NOTES"
    body = " ".join(f"{s} [S{i}]" for _, i, s in picked)
    prefix = "Hint (offline mode): look at this part of the notes: " if hint else "From the course notes: "
    return prefix + body


def clean_answer(text: str) -> str:
    """Model output -> answer. Drops a stray NOT_IN_NOTES after a real answer, greetings and sign-offs."""
    text = norm_citations(text.strip())
    if "NOT_IN_NOTES" in text and text.strip() != "NOT_IN_NOTES":
        rest = text.replace("NOT_IN_NOTES", "").strip()
        text = rest if len(rest) > 120 and re.search(r"\[S\d+\]", rest) else "NOT_IN_NOTES"
    text = re.sub(r"^(hello|hi|hey)[!,.]?[^\n]*?(teaching assistant|doubtloop)[^\n]*\n+", "", text, flags=re.I)
    text = re.sub(r"\n+(let me know|feel free|i hope this helps|happy studying)[^\n]*$", "", text, flags=re.I)
    return text.strip()


def answer(question: str, chunks, hint: bool = False, feedback: str = "", conversation: str = "",
           subject: str = None, style: str = "detailed", on_token=None) -> dict:
    if not chunks:
        return {"text": "NOT_IN_NOTES", "by": "rule"}
    user = f"Sources:\n{format_context(chunks)}\n\n"
    if conversation:
        user += ("Earlier in this chat (use only to understand what the student refers to; "
                 f"facts must still come from the sources):\n{conversation}\n\n")
    user += f"Student question: {question}"
    if feedback:
        user += f"\n\nA reviewer rejected your previous draft: {feedback}\nRewrite using only the sources."
    style = effective_style(question, style)
    if llm.is_online():
        try:
            system = HINT_SYSTEM if hint else answer_system(subject, style)
            return {"text": clean_answer(llm.chat(system, user, long=style == "exam", on_token=on_token)),
                    "by": "llm"}
        except llm.LLMError as exc:
            return {"text": _offline_answer(question, chunks, hint), "by": f"fallback ({str(exc)[:60]})"}
    return {"text": _offline_answer(question, chunks, hint), "by": "offline"}


WEB_SYSTEM = (
    "You are DoubtLoop, a teaching assistant for the university subject {subject}. The student's course "
    "notes cover this topic only briefly, so the student asked you to search the web. Use Google Search.\n"
    "- Explain it clearly and correctly in simple English, like a good teacher: a one-line definition, "
    "then the key points, causes/types/steps, a small real example, and how to avoid or apply it, as fits "
    "the topic.\n"
    "- Stay at university-course level for {subject}; prefer textbooks, universities and well-known "
    "references.\n"
    "- If the notes excerpt below says something, stay consistent with it and mention where the notes "
    "agree.\n"
    "- No greeting, no sign-off. Do not use [S1] citations. {length}")


def web_answer(question: str, note_text: str, subject: str, style: str = "detailed") -> dict:
    """Web research agent (runs only after the student approves): Gemini + Google Search grounding."""
    length = {"brief": "60-120 words.", "detailed": "180-350 words.",
              "exam": "350-600 words with bold headings."}.get(style, "180-350 words.")
    user = (f"Student question: {question}\n\nWhat the course notes say (may be short):\n"
            f"{note_text[:2500] or '(nothing)'}")
    try:
        text, sources = llm.web_chat(WEB_SYSTEM.format(subject=subject, length=length), user)
        return {"text": clean_answer(text), "sources": sources, "mode": "web"}
    except llm.LLMError as exc:
        print(f"[LLM] Gemini web search unavailable ({str(exc)[:120]}); trying the backup search", flush=True)
    from . import websearch
    results, provider = websearch.search(f"{question} {subject}")
    if results:
        block = "\n\n".join(f"[W{i}] {r['title']} ({r['url']})\n{r['text'][:900]}" for i, r in enumerate(results, 1))
        system = (WEB_SYSTEM.format(subject=subject, length=length)
                  .replace("Use Google Search.", "Use ONLY the numbered web results given, citing them as [W1], "
                           "[W2]. Ignore results that are off-topic or unreliable.")
                  .replace("Do not use [S1] citations.", "Do not use [S1] citations; use [W1]-style ones."))
        try:
            text = llm.chat(system, user + f"\n\nWeb results:\n{block}", long=style == "exam")
            return {"text": clean_answer(text), "sources": [{"title": r["title"] or r["url"], "url": r["url"]}
                                                             for r in results], "mode": "web",
                    "provider": provider}
        except llm.LLMError as exc:
            print(f"[LLM] could not write the web answer ({str(exc)[:120]})", flush=True)
    # No search available: Gemini explains from its own general knowledge instead
    # (the same knowledge the Gemini app uses), clearly labelled as such.
    print("[LLM] no web search available; using Gemini's general knowledge", flush=True)
    system = (WEB_SYSTEM.format(subject=subject, length=length)
              .replace("so the student asked you to search the web. Use Google Search.",
                       "so the student asked for a fuller explanation from your own general knowledge."))
    text = llm.chat(system, user, long=style == "exam")
    return {"text": clean_answer(text), "sources": [], "mode": "knowledge"}


REWRITE_SYSTEM = (
    "You help a search engine find the right part of a student's course notes. The first search found "
    "nothing good. Write 3 different short search queries (3-8 words each) for the same question, using "
    "the words course notes and syllabi usually use (e.g. 'lab experiments' -> 'list of practicals', "
    "'practical list', 'laboratory exercises'). Use the topics below as hints. "
    'JSON: {"queries": ["...", "...", "..."]}')


def rewrite_queries(question: str, topics=()) -> list:
    """Corrective retrieval: alternative wordings to search again when the first search is weak."""
    if not llm.is_online():
        return []
    try:
        out = llm.chat_json(REWRITE_SYSTEM, f"Question: {question}\nTopics in these notes: {', '.join(topics[:25])}")
        return [str(q).strip() for q in out.get("queries", []) if str(q).strip()][:3]
    except (llm.LLMError, ValueError, TypeError, AttributeError):
        return []


def pick_topic(question: str, chunks, doc_topics) -> str:
    """Dashboard topic for a doubt: the uploaded document's topic that best fits the question and the
    passage it was answered from (instead of 'General')."""
    if not chunks:
        return "General"
    text = (question + " " + chunks[0]["text"][:600]).lower()
    words = set(re.findall(r"[a-z0-9]{3,}", text))
    best, score = "", 0
    for t in doc_topics:
        tw = {w for w in re.findall(r"[a-z0-9]{3,}", t.lower()) if w not in {"and", "the", "for", "with"}}
        s = len(tw & words) + (2 if t.lower() in text else 0)
        if tw and s > score:
            best, score = t, s
    if best:
        return best
    sec = str(chunks[0]["meta"].get("section", "")).strip()
    return sec[:40] if sec and not sec.lower().startswith(("page", "-")) else "General"


COMPARE_Q = re.compile(r"\b(compare|comparison|differen\w*|distinguish|contrast|vs\.?|versus)\b", re.I)


def sub_queries(question: str) -> list:
    """Comparison doubts need the notes on BOTH ideas: 'Compare PRINCE2 and PMBOK' -> also search
    'PRINCE2' and 'PMBOK' separately (multi-query retrieval)."""
    if not COMPARE_Q.search(question):
        return []
    core = COMPARE_Q.sub(" ", question)
    core = re.sub(r"\b(between|in a table|in table|table|the|please|explain|give|with example|of|what|is|are)\b",
                  " ", core, flags=re.I)
    parts = [p.strip(" ?.,:") for p in re.split(r"\band\b|,|/|\bwith\b", core, flags=re.I)]
    return [p for p in parts if len(p) >= 3][:3]


# ------------------------------------------------------------------ critic agent
def quick_check(draft: str, chunks) -> dict:
    """Fast grounding check without an AI call: every [S#] must point to a real source, and the answer's
    content words must mostly come from the sources."""
    cited = {int(n) for n in re.findall(r"\[S(\d+)\]", draft)}
    valid = bool(cited) and all(1 <= n <= len(chunks) for n in cited)
    ctx = " ".join(c["text"].lower() for c in chunks)
    own = "\n".join(ln for ln in draft.splitlines() if "not from your notes" not in ln.lower())
    words = re.findall(r"[a-z]{5,}", re.sub(r"\[S\d+\]", "", own.lower()))
    support = sum(1 for w in words if w in ctx) / max(1, len(words))
    ok = valid and support >= 0.45
    return {"grounded": ok, "score": round(min(1.0, 0.55 + 0.5 * support), 2) if ok else round(support, 2),
            "issues": "" if ok else ("citations missing or invalid" if not valid else "low word support"),
            "by": "quick check"}


def critique(question: str, draft: str, chunks, quick: bool = False) -> dict:
    """Reflection step: is every claim in the draft supported by the sources?

    quick=True (strong retrieval match): a fast rule-based check first, and the AI critic only if that
    check fails. This is adaptive reflection: the expensive check runs only when it is needed."""
    if draft.strip() == "NOT_IN_NOTES":
        return {"grounded": False, "score": 0.0, "issues": "answer not in notes", "by": "rule"}
    if quick:
        q = quick_check(draft, chunks)
        if q["grounded"]:
            return q
    if llm.is_online():
        try:
            out = llm.chat_json(
                "You are a strict grader checking a teaching assistant's answer for hallucination.\n"
                "Check: (1) every claim is supported by the sources, (2) it answers the question as far as "
                "the sources allow. An answer that covers what the notes contain and clearly says which part "
                "is not covered is GOOD. Length and formatting are not issues. A line marked 'Example (added "
                "for clarity, not from your notes)' is allowed and must not be counted against the answer.\n"
                'JSON keys: "grounded" (true/false), "score" (0.0-1.0 confidence the answer is correct '
                'and supported), "issues" (short text, empty if none).',
                f"Sources:\n{format_context(chunks)}\n\nQuestion: {question}\n\nDraft answer:\n{draft}")
            return {"grounded": bool(out.get("grounded")), "score": float(out.get("score", 0)),
                    "issues": str(out.get("issues", "")), "by": "llm"}
        except (llm.LLMError, ValueError, TypeError):
            pass
    # offline: lexical support check
    ctx = " ".join(c["text"].lower() for c in chunks)
    words = [w for w in re.findall(r"[a-z]{5,}", re.sub(r"\[S\d+\]", "", draft.lower()))]
    support = sum(1 for w in words if w in ctx) / max(1, len(words))
    return {"grounded": support >= 0.6, "score": round(support, 2),
            "issues": "" if support >= 0.6 else "low lexical support", "by": "offline"}


# ------------------------------------------------------------------ insight agent
def faculty_insights(rows) -> str:
    """Summarise recent doubts into actionable points for the faculty."""
    if not rows:
        return "No doubts yet."
    from collections import Counter

    counts = Counter(r["topic"] for r in rows if r["category"] == "concept")
    sample = "\n".join(f"- ({r['topic']}) {r['question']}" for r in rows[:40])
    if llm.is_online():
        try:
            return llm.chat(
                "You help a professor plan the next lecture. From the student doubts, give exactly 3 "
                "bullet points: which topics confuse students most, the likely misconception, and one "
                "concrete thing to re-explain. Be specific and brief.",
                f"Doubt counts by topic: {dict(counts.most_common(8))}\nRecent doubts:\n{sample}")
        except llm.LLMError:
            pass
    top = counts.most_common(3)
    lines = [f"- **{t}**: {n} doubts this period, consider a 10-minute recap." for t, n in top]
    return "\n".join(lines) + "\n\n_(Offline summary. Add an API key for an AI-written analysis.)_"