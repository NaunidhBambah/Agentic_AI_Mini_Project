"""Evaluate DoubtLoop on a labelled test set.

Usage:  python eval/run_eval.py                 uses the test set the examiner agent wrote from YOUR library
                                                (eval/generated/test_set_auto.json), writing one if missing
        python eval/run_eval.py --generate      write a fresh test set first (after uploading new files)
        python eval/run_eval.py --tests test_set.json   a hand-written set instead
Output: printed summary + eval/results/report.md + eval/results/results.csv

Runs in a separate runtime folder, so the app's demo data is never touched.
"""
import json
import os
import argparse
import shutil
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--sleep", type=float, default=0, help="seconds to wait between questions (free-tier rate limits)")
parser.add_argument("--limit", type=int, default=0, help="only run the first N questions")
parser.add_argument("--tests", default="",
                    help="test file inside eval/ (default: the one generated for --subject)")
parser.add_argument("--subject", default="", help="subject to evaluate (default: COURSE_NAME in .env)")
parser.add_argument("--generate", action="store_true", help="let the examiner agent write a new test set first")
ARGS = parser.parse_args()

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "eval" / "results" / "_runtime"
shutil.rmtree(RUNTIME, ignore_errors=True)
os.environ["DOUBTLOOP_RUNTIME"] = str(RUNTIME)
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from doubtloop import llm, service, testgen  # noqa: E402
from doubtloop.embeddings import get_embedding  # noqa: E402

EXPECTED = {  # test type -> statuses that count as correct behaviour
    "answerable": {"answered", "verified"},
    "not_in_notes": {"escalated"},
    "assignment": {"hint"},
    "off_topic": {"refused"},
    "admin": {"escalated"},
    "blocked": {"blocked"},
    "pii": {"answered", "verified"},
}
SHOULD_ESCALATE = {"not_in_notes", "admin"}


def main():
    print(f"LLM: {llm.mode_label()} | embeddings: {get_embedding()[1]}")
    service.startup(seed=False)
    subject = ARGS.subject or service.knowledge.DEFAULT_SUBJECT
    path = ROOT / "eval" / ARGS.tests if ARGS.tests else testgen.test_file(subject)
    print(f"Subject: {subject} | test set: {path.relative_to(ROOT)}")
    if ARGS.generate or (not ARGS.tests and not path.exists()):
        print("Examiner agent is writing a test set from the course library...")
        result = testgen.generate(on_step=lambda m: print("  " + m), subject=subject)
        if not result["tests"]:
            sys.exit(result["summary"])
        testgen.save(result)
        print(result["summary"])
    tests = [t for t in json.loads(path.read_text(encoding="utf-8")) if t.get("include", True)]
    if ARGS.limit:
        tests = tests[:ARGS.limit]
    rows = []
    for n, t in enumerate(tests, start=1):
        t0 = time.perf_counter()
        d = service.ask(f"eval-{t['id']}", t["question"], auto_confirm=True, subject=subject)
        ms = int((time.perf_counter() - t0) * 1000)
        ans = (d["answer"] or "").lower()
        kws = t.get("keywords", [])
        kw_cov = sum(1 for k in kws if k.lower() in ans) / len(kws) if kws else None
        rows.append({
            "id": t["id"], "type": t["type"], "question": t["question"], "status": d["status"],
            "correct_behaviour": d["status"] in EXPECTED[t["type"]],
            "escalated": d["status"] == "escalated", "should_escalate": t["type"] in SHOULD_ESCALATE,
            "keyword_coverage": kw_cov, "cited": "[s" in ans,
            "pii_masked": ("MASKED" in d["question"]) if t["type"] == "pii" else None,
            "confidence": d["confidence"], "latency_ms": ms,
            "trajectory": " > ".join(s["node"] for s in d["trace"]),
        })
        print(f"{t['id']:4} {t['type']:13} -> {d['status']:10} {'OK ' if rows[-1]['correct_behaviour'] else 'XX '} {ms} ms")
        print(f"PROGRESS {n}/{len(tests)}", flush=True)
        time.sleep(ARGS.sleep)

    df = pd.DataFrame(rows)
    ans = df[df["type"] == "answerable"]
    tp = ((df.escalated) & (df.should_escalate)).sum()
    fp = ((df.escalated) & (~df.should_escalate)).sum()
    fn = ((~df.escalated) & (df.should_escalate)).sum()
    guard = df[df["type"].isin(["assignment", "off_topic", "blocked", "pii"])]
    metrics = {
        f"Behaviour accuracy (all {len(df)})": f"{df.correct_behaviour.mean() * 100:.1f}%",
        "Answerable: answered (task completion)": f"{ans.correct_behaviour.mean() * 100:.1f}%",
        "Answerable: keyword coverage": f"{ans.keyword_coverage.mean() * 100:.1f}%",
        "Answerable: citation rate": f"{ans[ans.correct_behaviour].cited.mean() * 100:.1f}%",
        "Escalation precision": f"{tp / max(1, tp + fp) * 100:.1f}%",
        "Escalation recall": f"{tp / max(1, tp + fn) * 100:.1f}%",
        "Guardrail accuracy (assignment/off-topic/attack/PII)": f"{guard.correct_behaviour.mean() * 100:.1f}%",
        "PII masked": (f"{df[df.type == 'pii'].pii_masked.mean() * 100:.0f}%" if (df.type == "pii").any()
                       else "not tested"),
        "Average latency": f"{df.latency_ms.mean():.0f} ms",
    }
    per_type = df.groupby("type").correct_behaviour.mean().mul(100).round(1)

    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    lines = ["# DoubtLoop evaluation report", "", f"- Subject: {subject}",
             f"- Test set: {path.relative_to(ROOT)}",
             f"- LLM: {llm.mode_label()}", f"- Embeddings: {get_embedding()[1]}",
             f"- Test questions: {len(df)}", "", "## Metrics", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in metrics.items()]
    lines += ["", "## Accuracy by question type", "", "| Type | Correct behaviour |", "|---|---|"]
    lines += [f"| {k} | {v}% |" for k, v in per_type.items()]
    wrong = df[~df.correct_behaviour]
    if len(wrong):
        lines += ["", "## Failures (for error analysis)", "", "| ID | Type | Got | Question |", "|---|---|---|---|"]
        lines += [f"| {r.id} | {r.type} | {r.status} | {r.question} |" for r in wrong.itertuples()]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(f"{k:55} {v}" for k, v in metrics.items()))
    print(f"\nSaved: {out / 'report.md'} and results.csv")


if __name__ == "__main__":
    main()