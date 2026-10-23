"""Guardrails: deterministic Python checks that run before and after the LLM.

Input guardrails : PII masking, prompt-injection block, abuse block, length limit
Output guardrails: citation required, hint-only for graded work, PII re-mask
Each check returns events so the UI can show a visible "Guardrail" badge.
"""
import re

from . import config

PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("phone", re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b")),
    ("id_number", re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b|\b\d{9,16}\b")),
]

INJECTION = re.compile(
    r"ignore (all |the )?(previous|above|prior) (instructions|rules)|you are now|"
    r"system prompt|developer mode|jailbreak|pretend (to be|you are)|act as (an? )?unrestricted",
    re.I)

ABUSE = re.compile(r"\b(idiot|stupid|moron|shut up|f+u+c+k+|bitch|bastard)\b", re.I)

# Asking for a complete solution is treated as graded work (hint-only), even without a match.
SOLUTION_REQUEST = re.compile(
    r"\b(full|complete|entire|whole)\s+(code|solution|program|implementation|answer)\b|"
    r"\bsolve\s+(this|the|my)?\s*(quiz|assignment|lab|exam)|\b(lab|assignment)\s*\d+\s+(code|solution)", re.I)


TASK_REQUEST = re.compile(r"\b(write|implement|code|build|create|solve|provide|develop|program|"
                          r"give me|do my|complete|submit|answer (this|the))\b", re.I)


def is_task_request(text: str) -> bool:
    """'Write/implement/solve X' style requests (vs. conceptual 'what/why/how' questions)."""
    return bool(TASK_REQUEST.search(text))


def is_solution_request(text: str) -> bool:
    return bool(SOLUTION_REQUEST.search(text))


CITATION = re.compile(r"\[S\d+\]")
CODE_BLOCK = re.compile(r"```.*?```", re.S)


def event(name: str, detail: str, action: str) -> dict:
    return {"guardrail": name, "detail": detail, "action": action}


def mask_pii(text: str):
    events = []
    for label, pat in PII_PATTERNS:
        if pat.search(text):
            text = pat.sub(f"[{label.upper()} MASKED]", text)
            events.append(event("pii_masking", f"{label} removed before storage", "masked"))
    return text, events


def check_input(text: str):
    """Returns (cleaned_text, events, block_reason_or_None)."""
    events = []
    text = (text or "").strip()
    if not text:
        return text, events, "Please type a question."
    if len(text) > config.MAX_QUESTION_CHARS:
        text = text[: config.MAX_QUESTION_CHARS]
        events.append(event("length_limit", f"question trimmed to {config.MAX_QUESTION_CHARS} chars", "trimmed"))
    text, pii = mask_pii(text)
    events += pii
    if INJECTION.search(text):
        events.append(event("prompt_injection", "attempt to override the assistant's rules", "blocked"))
        return text, events, ("This looks like an attempt to change my instructions. "
                              "I can only help with doubts about the course.")
    if ABUSE.search(text):
        events.append(event("abuse_filter", "abusive language", "blocked"))
        return text, events, "Let's keep it respectful. Please rephrase your doubt."
    return text, events, None


def check_output(answer: str, mode: str, has_sources: bool):
    """Returns (final_answer, events, ok). ok=False means the answer must not be shown."""
    events = []
    answer, pii = mask_pii(answer)
    events += pii

    if mode == "hint":
        long_code = [b for b in CODE_BLOCK.findall(answer) if b.count("\n") > 6]
        if long_code:
            for b in long_code:
                answer = answer.replace(b, "_(code removed: write this part yourself)_")
            events.append(event("academic_integrity", "full code solution removed from hint", "redacted"))
        if len(answer) > 1200:
            answer = answer[:1200].rsplit(" ", 1)[0] + " ..."
            events.append(event("academic_integrity", "hint shortened", "trimmed"))
        return answer, events, True

    if mode == "notes" and has_sources and not CITATION.search(answer):
        events.append(event("citation_required", "answer had no citation to the notes", "withheld"))
        return answer, events, False
    return answer, events, True
