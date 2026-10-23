"""LLM access through LangChain chat models: Gemini (default), OpenAI, Claude or Groq.

If no API key is configured the app still runs in OFFLINE mode: answers are built
extractively from the retrieved notes, so the whole pipeline can be demonstrated.
"""
import json
import re
import time
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage

from . import config


class LLMError(RuntimeError):
    pass


def _need(package: str, install: str):
    try:
        return __import__(package)
    except ImportError as exc:
        raise LLMError(f"Missing package. Run: pip install {install}") from exc


def _gemini(model: str, max_tokens: int, retries: int):
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = {"thinking_level": config.GEMINI_THINKING} if config.GEMINI_THINKING else {}
    return ChatGoogleGenerativeAI(model=model, google_api_key=config.GOOGLE_API_KEY, temperature=0.1,
                                  max_output_tokens=max_tokens, max_retries=retries, timeout=40, **kwargs)


@lru_cache(maxsize=4)
def _model(fast: bool = False, long: bool = False):
    """Build the chat model for the provider chosen in .env (all via LangChain).

    fast=True is used for short internal steps (router, critic, follow-up rewriting, cataloguing):
    same model, smaller output budget, so those steps return quickly.
    """
    p = config.LLM_PROVIDER
    tokens = 600 if fast else (4000 if long else 2500)   # Gemini counts its thinking in this budget too
    if p == "gemini" and config.GOOGLE_API_KEY:
        _need("langchain_google_genai", "langchain-google-genai")
        return _gemini((config.GEMINI_FAST_MODEL or config.GEMINI_MODEL) if fast else config.GEMINI_MODEL,
                       tokens, 0 if fast else 1)
    if p == "openai" and config.OPENAI_API_KEY:
        _need("langchain_openai", "langchain-openai")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.OPENAI_MODEL, api_key=config.OPENAI_API_KEY,
                          max_retries=3, timeout=60, max_tokens=tokens)
    if p == "anthropic" and config.ANTHROPIC_API_KEY:
        _need("langchain_anthropic", "langchain-anthropic")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.ANTHROPIC_MODEL, api_key=config.ANTHROPIC_API_KEY,
                             temperature=0.1, max_tokens=tokens, max_retries=3, timeout=60)
    if p == "groq" and config.GROQ_API_KEY:
        _need("langchain_groq", "langchain-groq")
        from langchain_groq import ChatGroq

        kwargs = {"reasoning_effort": "low"} if "gpt-oss" in config.GROQ_MODEL else {}
        return ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0.1,
                        max_tokens=tokens, max_retries=6, timeout=60, **kwargs)
    return None


@lru_cache(maxsize=1)
def _fallback_model():
    """A lighter Gemini model to use when the main one is busy (only for the Gemini provider)."""
    if config.LLM_PROVIDER != "gemini" or not config.GOOGLE_API_KEY or not config.GEMINI_FALLBACK_MODEL:
        return None
    return _gemini(config.GEMINI_FALLBACK_MODEL, 4000, 1)


# Errors after which the lighter fallback model is tried (busy, rate-limited or too slow)
_BUSY = ("503", "UNAVAILABLE", "overloaded", "high demand", "429", "RESOURCE_EXHAUSTED", "rate limit",
         "504", "DEADLINE_EXCEEDED", "timed out", "timeout")


def is_online() -> bool:
    return _model() is not None


def mode_label() -> str:
    if not is_online():
        return "Offline demo mode (no API key)"
    names = {"gemini": ("Gemini", config.GEMINI_MODEL), "openai": ("OpenAI", config.OPENAI_MODEL),
             "anthropic": ("Claude", config.ANTHROPIC_MODEL), "groq": ("Groq", config.GROQ_MODEL)}
    brand, model = names[config.LLM_PROVIDER]
    return f"{brand} · {model}"


# A model that answered "429 quota / 503 busy" is skipped for a while instead of being retried with long
# waits (the retries were the main reason answers took 40-110 s on a free key).
_COOLDOWN = {}
COOLDOWN_S = 90


def _name(model) -> str:
    return getattr(model, "model", None) or getattr(model, "model_name", None) or "model"


def _is_busy(exc) -> bool:
    return any(b.lower() in str(exc).lower() for b in _BUSY)


def _text(content) -> str:
    if isinstance(content, list):  # some providers return content blocks
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def _candidates(fast: bool, long: bool) -> list:
    cands = [_model(fast, long and not fast)]
    if fast and _model(False) is not cands[0]:
        cands.append(_model(False))                 # light model failed: try the main one
    fb = _fallback_model()
    if fb is not None and all(_name(fb) != _name(c) for c in cands):
        cands.append(fb)
    live = [c for c in cands if _COOLDOWN.get(_name(c), 0) < time.time()]
    return live or cands[-1:]


def chat(system: str, user: str, fast: bool = False, long: bool = False, on_token=None) -> str:
    """One AI call. long=True: bigger output budget. on_token(text_so_far): stream the answer live."""
    if _model() is None:
        raise LLMError("No LLM configured")
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    errors = []
    for model in _candidates(fast, long):
        name, t0 = _name(model), time.perf_counter()
        try:
            if on_token:
                parts = []
                for chunk in model.stream(messages):
                    piece = _text(chunk.content)
                    if piece:
                        parts.append(piece)
                        on_token("".join(parts))
                text = "".join(parts)
            else:
                text = _text(model.invoke(messages).content)
            # one line per AI call in the terminal, so slow steps are easy to spot
            print(f"[LLM] {name} {time.perf_counter() - t0:.1f}s ({len(system) + len(user)} chars in)"
                  + (" [backup]" if errors else ""), flush=True)
            return text.strip()
        except Exception as exc:  # network, rate limit, bad key ...
            errors.append(f"{name}: {str(exc)[:300]}")
            if _is_busy(exc):
                _COOLDOWN[name] = time.time() + COOLDOWN_S
                print(f"[LLM] {name} is rate-limited/busy -> using the backup model for {COOLDOWN_S}s", flush=True)
    raise LLMError(" | ".join(errors))


def friendly_error(exc) -> str:
    """Short message for students instead of a raw API error."""
    s = str(exc)
    if "RESOURCE_EXHAUSTED" in s or "429" in s or "quota" in s.lower():
        return ("The free Gemini quota is used up for now (too many requests). Wait a minute and try again; "
                "if it keeps happening the daily limit is reached, so try again tomorrow or use a key with billing.")
    if "503" in s or "UNAVAILABLE" in s or "overloaded" in s.lower():
        return "Gemini is busy right now. Try again in a moment."
    return s[:300]


def web_available() -> bool:
    """Web search uses Gemini's built-in Google Search grounding (same API key)."""
    return config.LLM_PROVIDER == "gemini" and bool(config.GOOGLE_API_KEY)


def web_chat(system: str, user: str) -> tuple:
    """Answer with Google Search grounding. Returns (text, [{title, url}])."""
    if not web_available():
        raise LLMError("Web search needs the Gemini provider (LLM_PROVIDER=gemini with GOOGLE_API_KEY).")
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    errors = []
    for name in dict.fromkeys([config.GEMINI_MODEL, config.GEMINI_FALLBACK_MODEL]):
        if not name or _COOLDOWN.get(f"web:{name}", 0) > time.time():
            continue
        t0 = time.perf_counter()
        try:
            reply = _gemini(name, 3000, 0).bind_tools([{"google_search": {}}]).invoke(messages)
        except Exception as exc:
            errors.append(f"{name}: {str(exc)[:200]}")
            if _is_busy(exc):
                _COOLDOWN[f"web:{name}"] = time.time() + COOLDOWN_S
            continue
        print(f"[LLM] web search {name} {time.perf_counter() - t0:.1f}s", flush=True)
        meta = (reply.response_metadata or {}).get("grounding_metadata") or {}
        sources, seen = [], set()
        for ch in meta.get("grounding_chunks") or []:
            web = (ch or {}).get("web") or {}
            url, title = web.get("uri"), web.get("title") or web.get("domain") or "web page"
            if url and title not in seen:
                seen.add(title)
                sources.append({"title": title, "url": url})
        return _text(reply.content).strip(), sources[:6]
    raise LLMError(" | ".join(errors) or "web search is cooling down after a quota error")


def chat_json(system: str, user: str, fast: bool = True) -> dict:
    """Ask for JSON and parse it robustly (models sometimes wrap it in ``` fences)."""
    text = chat(system + "\nReply with ONE valid JSON object only. No markdown.", user, fast=fast)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise LLMError(f"Model did not return JSON: {text[:200]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f"Bad JSON from model: {text[:200]}") from exc