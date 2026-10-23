"""Learning-curve experiment: does the escalation rate fall as the TA answers doubts?

Three simulated "weeks". Each week students ask 10 course questions (answerable from notes)
plus 10 questions the notes do NOT cover. Week 2 and 3 use reworded versions of the same
uncovered questions. After each week a simulated TA answers whatever was escalated.

Usage:  python eval/learning_curve.py
Output: eval/results/learning_curve.png and learning_curve.csv
"""
import os
import argparse
import shutil
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--no-llm", action="store_true",
                    help="run without LLM calls (saves free-tier quota; memory reuse is embedding-based)")
parser.add_argument("--sleep", type=float, default=0, help="seconds to wait between questions")
parser.add_argument("--builtin", action="store_true",
                    help="use the built-in Agentic AI questions instead of the generated test set")
parser.add_argument("--subject", default="", help="subject to use (default: COURSE_NAME in .env)")
ARGS = parser.parse_args()
if ARGS.no_llm:
    os.environ["LLM_PROVIDER"] = "offline"

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "eval" / "results" / "_runtime_lc"
shutil.rmtree(RUNTIME, ignore_errors=True)
os.environ["DOUBTLOOP_RUNTIME"] = str(RUNTIME)
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from doubtloop import knowledge, llm, service, store, testgen  # noqa: E402

SUBJECT = ARGS.subject or knowledge.DEFAULT_SUBJECT
from doubtloop.embeddings import get_embedding  # noqa: E402

# (week-1 wording, week-2 wording, week-3 wording, TA reference answer)
UNCOVERED = [
    ("What is MCP (Model Context Protocol)?", "Can you explain the MCP Model Context Protocol?",
     "What does MCP mean in agentic AI?",
     "MCP is an open standard that lets AI apps connect to tools and data through MCP servers, building on function calling."),
    ("What is GraphRAG?", "How does GraphRAG work?", "Explain GraphRAG retrieval.",
     "GraphRAG builds a knowledge graph of entities and relations from documents and retrieves over that graph instead of only plain chunks."),
    ("What is LoRA fine-tuning?", "Explain LoRA for fine-tuning LLMs.", "How does LoRA fine-tuning work?",
     "LoRA fine-tunes a model by training small low-rank adapter matrices while the original weights stay frozen."),
    ("What is DSPy?", "What is the DSPy framework used for?", "Explain DSPy prompt optimisation.",
     "DSPy is a framework that treats prompts as programs and optimises them automatically against a metric."),
    ("What is the A2A protocol?", "Explain the A2A Agent2Agent protocol.", "How does A2A let agents communicate?",
     "A2A is a protocol that lets agents from different vendors discover each other and exchange tasks and messages."),
    ("What is a GGUF model file?", "Explain the GGUF format.", "What is GGUF used for in local LLMs?",
     "GGUF is a file format for quantised models used by llama.cpp and Ollama to run LLMs locally."),
    ("What is vLLM?", "What is vLLM used for?", "How does vLLM serve models faster?",
     "vLLM is an open-source inference server that uses PagedAttention to serve LLMs with high throughput."),
    ("What is RLHF?", "Explain RLHF training.", "How does RLHF align an LLM?",
     "RLHF trains a reward model from human preference rankings and then optimises the LLM against that reward."),
    ("What is MoE (mixture of experts)?", "Explain MoE mixture of experts models.", "How does an MoE model route tokens?",
     "A mixture-of-experts model has many expert sub-networks and a router that activates only a few per token."),
    ("What is a KV cache in LLMs?", "Explain the KV cache.", "Why does the KV cache speed up LLM generation?",
     "The KV cache stores attention keys and values of previous tokens so they are not recomputed for each new token."),
]
COVERED = [
    ("What is MCP (Model Context Protocol)?", "Can you explain the Model Context Protocol (MCP)?",
     "What does MCP mean in agentic AI?",
     "MCP is an open standard that lets AI apps connect to tools and data through MCP servers, building on function calling."),
    ("What is GraphRAG?", "How does GraphRAG work?", "Explain GraphRAG and knowledge graph retrieval.",
     "GraphRAG builds a knowledge graph of entities and relations from documents and retrieves over that graph instead of only plain chunks."),
    ("What is LoRA fine-tuning?", "Explain LoRA for fine-tuning LLMs.", "How does LoRA fine-tuning work?",
     "LoRA fine-tunes a model by training small low-rank adapter matrices while the original weights stay frozen."),
    ("What is tree-of-thoughts prompting?", "Explain tree of thoughts prompting.", "How does tree-of-thoughts reasoning work?",
     "Tree-of-thoughts explores several reasoning branches, evaluates them and keeps the most promising ones."),
    ("What is DSPy?", "What is the DSPy framework used for?", "Explain DSPy prompt optimisation.",
     "DSPy is a framework that treats prompts as programs and optimises them automatically against a metric."),
    ("What is the A2A protocol?", "Explain the Agent2Agent (A2A) protocol.", "How does A2A let agents communicate?",
     "A2A is a protocol that lets agents from different vendors discover each other and exchange tasks and messages."),
    ("What is speculative decoding?", "Explain speculative decoding in LLMs.", "How does speculative decoding speed up inference?",
     "A small draft model proposes tokens and the large model verifies them in parallel, which speeds up generation."),
    ("What is a GGUF model file?", "Explain the GGUF format.", "What is GGUF used for in local LLMs?",
     "GGUF is a file format for quantised models used by llama.cpp and Ollama to run LLMs locally."),
    ("What is Ollama?", "Explain what Ollama does.", "How is Ollama used to run local models?",
     "Ollama is a tool to download and run open LLMs locally with a simple command line and API."),
    ("What is semantic caching for LLMs?", "Explain semantic caching for LLM apps.", "How does a semantic cache reduce LLM cost?",
     "A semantic cache stores previous answers and returns them when a new query is similar enough, saving LLM calls."),
]
COVERED = [
    "What is episodic memory?", "Why use chunk overlap?", "What is cosine similarity?",
    "Explain the ReAct pattern.", "What is function calling?", "What is ChromaDB?",
    "What is the orchestrator-worker pattern?", "What are output guardrails?",
    "What is a model card?", "What is Plan-and-Execute?",
]


def load_generated():
    """Use the examiner agent's test set: its 'not in notes' questions (with 2 rewordings and a
    reference answer each) play the uncovered doubts, its answerable questions the covered ones."""
    path = testgen.test_file(SUBJECT)
    if ARGS.builtin or not path.exists():
        return None
    import json
    tests = [t for t in json.loads(path.read_text(encoding="utf-8")) if t.get("include", True)]
    unc = [(t["question"], *(t.get("paraphrases", []) + [t["question"]] * 2)[:2],
            t.get("ta_answer") or "See the reference text.")
           for t in tests if t["type"] == "not_in_notes"]
    cov = [t["question"] for t in tests if t["type"] == "answerable"]
    if not unc:
        print("Generated set has no 'not in notes' questions (needs the AI model); using built-in set.")
        return None
    print(f"Using generated test set: {len(unc)} uncovered + {len(cov)} covered questions")
    return unc, cov


def main():
    global UNCOVERED, COVERED
    print(f"LLM: {llm.mode_label()} | embeddings: {get_embedding()[1]}")
    service.startup(seed=False)
    gen = load_generated()
    if gen:
        UNCOVERED, COVERED = gen
    if len(UNCOVERED) < 4:
        print(f"Note: only {len(UNCOVERED)} 'not in notes' questions. With so few, one doubt changes the rate by "
              "a lot; keep 4-6 ticked in Self-test for a reliable curve.")
    rows, log = [], []
    unc_text = {w: i for i, u in enumerate(UNCOVERED) for w in u[:3]}
    for week in range(3):
        asked = []
        for i, q in enumerate([u[week] for u in UNCOVERED] + COVERED):
            d = service.ask(f"w{week}-student{i}", q, auto_confirm=True, subject=SUBJECT)
            asked.append((q in unc_text, d))
            log.append({"week": week + 1, "kind": "new topic" if q in unc_text else "in notes", "question": q,
                        "status": d["status"], "answered_by": d.get("answered_by", "")})
            time.sleep(ARGS.sleep)
        new = [d for u, d in asked if u]
        old = [d for u, d in asked if not u]
        esc_new = sum(d["status"] == "escalated" for d in new)
        esc_old = sum(d["status"] == "escalated" for d in old)
        rows.append({"week": f"Week {week + 1}", "questions": len(asked),
                     "new_topic_doubts": len(new), "new_topic_escalated": esc_new,
                     "new_topic_escalation_%": round(esc_new / max(1, len(new)) * 100, 1),
                     "answered_from_TA_memory": sum(d["status"] == "verified" for d in new),
                     "in_notes_escalated": esc_old,
                     "overall_escalation_%": round((esc_new + esc_old) / len(asked) * 100, 1)})
        print(rows[-1])
        # Simulated TA: answers (and saves) only the genuinely uncovered doubts. Anything else that was
        # escalated is answered for that student only, so a generic reply never pollutes the memory.
        for d in store.list_doubts(status=["escalated"]):
            i = unc_text.get(d["question"])
            if i is not None:
                service.ta_reply(d["id"], UNCOVERED[i][3], "Sim TA")
            else:
                service.ta_reply(d["id"], "Please see the lecture slides for this topic.", "Sim TA",
                                 save_to_memory=False)

    df = pd.DataFrame(rows)
    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "learning_curve.csv", index=False)
    pd.DataFrame(log).to_csv(out / "learning_curve_log.csv", index=False)

    y = df["new_topic_escalation_%"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(df["week"], y, marker="o", linewidth=2.5, color="#534AB7",
            label=f"Doubts not in the notes ({len(UNCOVERED)} topics, reworded each week)")
    ax.plot(df["week"], df["overall_escalation_%"], marker="s", linewidth=1.5, linestyle="--", color="#9AA0B4",
            label="All doubts")
    for x, v in zip(df["week"], y):
        ax.annotate(f"{v:.0f}%", (x, v), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_ylabel("Escalated to a TA (%)")
    ax.set_ylim(0, 110)
    falling = y.iloc[-1] < y.iloc[0]
    ax.set_title("Escalations fall as DoubtLoop learns from TA answers" if falling
                 else "Escalation to a TA by week")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=160)
    lg = pd.DataFrame(log)
    first = lg[(lg.kind == "new topic") & (lg.week == 1)].reset_index(drop=True)
    reached = [i for i, r in first.iterrows() if r.status == "escalated"]
    later = lg[(lg.kind == "new topic") & (lg.week > 1)].reset_index(drop=True)
    reused = sum(later.iloc[w * len(first) + i].status == "verified" for w in range(2) for i in reached
                 if w * len(first) + i < len(later))
    print(f"Saved: {out / 'learning_curve.png'} (+ learning_curve_log.csv with every question)")
    print(f"TA-answered topics from week 1: {len(reached)}; their rewordings in weeks 2-3 answered from TA "
          f"memory: {reused} of {2 * len(reached)}. Topics that never reached the TA (answered or refused "
          "by the agent) can't be learned, which is what makes the curve uneven.")


if __name__ == "__main__":
    main()