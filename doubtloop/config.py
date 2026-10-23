"""Central configuration. Every value can be overridden in the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _get(name: str, default: str) -> str:
    """Read from environment first, then Streamlit secrets (for cloud deploy)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st  # only available when running inside Streamlit

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


# ---------- Course ----------
# Shown in the sidebar, PDFs and refusal messages. Answers always come from whatever files are uploaded.
COURSE_NAME = _get("COURSE_NAME", "Agentic AI (Sem VII)")

# ---------- LLM ----------
# gemini (recommended, free tier) | openai | anthropic | groq | offline
LLM_PROVIDER = _get("LLM_PROVIDER", "gemini").lower()
GOOGLE_API_KEY = _get("GOOGLE_API_KEY", "")
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-3.8-flash")
# Used automatically when the main model is overloaded (503) or rate-limited (429)
GEMINI_FALLBACK_MODEL = _get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
# How much Gemini "thinks" before answering: low = fast (recommended), medium/high = slower.
# Gemini 3 Flash defaults to high when this is not sent, which makes every step slow.
GEMINI_THINKING = _get("GEMINI_THINKING", "low").lower().strip()
# Quick internal checks (router, critic, follow-up rewriting, cataloguing) use this lighter, faster
# model; student-facing answers and summaries still use GEMINI_MODEL. Set equal to GEMINI_MODEL to disable.
GEMINI_FAST_MODEL = _get("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")
OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-5-mini")
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = _get("ANTHROPIC_MODEL", "claude-haiku-5-5")
GROQ_API_KEY = _get("GROQ_API_KEY", "")
GROQ_MODEL = _get("GROQ_MODEL", "openai/gpt-oss-120b")

# ---------- Embeddings ----------
# "default" = ChromaDB's built-in all-MiniLM-L6-v2 (ONNX, ~80 MB, downloaded once)
# "simple"  = offline hashing embeddings (no download, lower quality)
EMBEDDINGS = _get("EMBEDDINGS", "default").lower()

# ---------- Paths ----------
DATA_DIR = ROOT / "data"
NOTES_DIR = DATA_DIR / "course_notes"
SEED_DIR = DATA_DIR / "seed"
# runtime/ is created automatically and is safe to delete (it is rebuilt on start).
# Evaluation scripts point this to a separate folder so they never touch the demo data.
RUNTIME_DIR = Path(os.getenv("DOUBTLOOP_RUNTIME", str(ROOT / "runtime")))
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
NOTES_DIR.mkdir(parents=True, exist_ok=True)       # created if missing (e.g. after copying to a new laptop)
DB_PATH = RUNTIME_DIR / "doubtloop.db"
CHECKPOINT_PATH = RUNTIME_DIR / "checkpoints.db"
CHROMA_DIR = RUNTIME_DIR / "chroma"

# ---------- Agent behaviour ----------
TOP_K = int(_get("TOP_K", "4"))                  # chunks given to the answer agent
CHUNK_SIZE = int(_get("CHUNK_SIZE", "700"))       # characters
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "120"))
MAX_CRITIC_RETRIES = 1                            # loop guard for the reflection cycle
# auto = AI critic only when the notes match weakly (fast); always = AI critic on every answer; off = rules only
CRITIC_MODE = _get("CRITIC_MODE", "auto").lower().strip()
# Optional backup web search when Gemini's Google Search quota is used up (free key at tavily.com)
TAVILY_API_KEY = _get("TAVILY_API_KEY", "")
MIN_RETRIEVAL_SIM = float(_get("MIN_RETRIEVAL_SIM", "0.30"))
CONFIDENCE_THRESHOLD = float(_get("CONFIDENCE_THRESHOLD", "0.55"))
TA_MATCH_SIM = float(_get("TA_MATCH_SIM", "0.68"))           # reuse a TA answer above this
ASSIGNMENT_MATCH_SIM = float(_get("ASSIGNMENT_MATCH_SIM", "0.70"))
MAX_QUESTION_CHARS = 800