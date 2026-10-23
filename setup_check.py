"""Run this once after installing: python setup_check.py

Checks Python, packages, your API key, the embedding model, and runs one test doubt.
"""
import os
import sys
from pathlib import Path

# Use a scratch runtime folder so this check doesn't add records to the app's dashboard.
os.environ["DOUBTLOOP_RUNTIME"] = str(Path(__file__).parent / "eval" / "results" / "_runtime_check")

print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info >= (3, 10) else "-> please use Python 3.11 or 3.12")

missing = []
for mod in ["langgraph", "langchain_core", "chromadb", "streamlit", "pypdf", "pandas", "dotenv"]:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit(f"Missing packages: {missing}\nRun: pip install -r requirements.txt")
print("Packages OK")

from doubtloop import config, llm, service  # noqa: E402
from doubtloop.embeddings import get_embedding  # noqa: E402

print(f"LLM provider: {config.LLM_PROVIDER} -> {llm.mode_label()}")
if llm.is_online():
    try:
        print("LLM test reply:", llm.chat("Reply with exactly one word.", "Say OK")[:40])
    except llm.LLMError as exc:
        print("LLM ERROR:", exc)
        print("-> Check the API key in .env, your internet connection, or the model name.")
else:
    print(f"No API key found for LLM_PROVIDER={config.LLM_PROVIDER}: the app will run in offline demo mode.")
    print("-> For Gemini add GOOGLE_API_KEY to .env and run: pip install langchain-google-genai")

_, backend = get_embedding()
print("Embeddings:", "all-MiniLM-L6-v2 (good)" if backend == "minilm" else "offline hashing fallback (lower quality)")

info = service.startup()
print(f"Indexed {info['chunks']} note chunks | TA memory {info['ta_memory']} | doubts {info['doubts']}")
d = service.ask("setup-check", "What is the difference between episodic and semantic memory?", auto_confirm=True)
print(f"Test doubt -> status: {d['status']}, confidence: {d['confidence']:.2f}")
print("Answer preview:", (d["answer"] or "")[:200].replace("\n", " "))
if d["status"] in ("answered", "verified"):
    print("\nAll good. Start the app with:  streamlit run app.py")
else:
    print(f"\nWARNING: expected 'answered' but got '{d['status']}'. Send this output for a fix.")