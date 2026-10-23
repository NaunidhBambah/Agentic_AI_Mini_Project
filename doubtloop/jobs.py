"""Background jobs for long faculty tasks (writing a test set, running the evaluation).

Streamlit re-runs the page on every click, which used to cancel a task that ran inside the page.
These jobs run in a background thread instead, so the faculty member can switch tabs, roles or even
refresh the browser while they continue. Only one job of each kind runs at a time, because they share
one API key and free-tier rate limits.
"""
import re
import subprocess
import sys
import threading
import time
import os

from . import config

_JOBS = {}
_LOCK = threading.Lock()


def get(kind: str) -> dict:
    return _JOBS.get(kind)


def running(kind: str) -> bool:
    j = _JOBS.get(kind)
    return bool(j and j["state"] == "running")


def _new(kind):
    job = {"kind": kind, "state": "running", "steps": [], "result": None, "error": "",
           "started": time.time(), "finished": None, "progress": 0.0}
    _JOBS[kind] = job
    return job


def _run(job, fn):
    try:
        job["result"] = fn(job)
        job["state"] = "done"
    except Exception as exc:          # shown in the UI instead of crashing the thread silently
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["state"] = "error"
    job["finished"] = time.time()


def start(kind: str, fn) -> bool:
    """Start fn(job) in the background. Returns False if a job of this kind is already running."""
    with _LOCK:
        if running(kind):
            return False
        job = _new(kind)
    threading.Thread(target=_run, args=(job, fn), daemon=True, name=f"doubtloop-{kind}").start()
    return True


def start_generation(n_answerable: int, n_unseen: int, files: list, subject: str = None) -> bool:
    from . import testgen

    def work(job):
        return testgen.generate(n_answerable, n_unseen, files=files, on_step=job["steps"].append,
                                subject=subject)
    return start("generate", work)


def start_evaluation(gap: float, subject: str = None) -> bool:
    """Runs eval/run_eval.py as a separate process (its own sandbox data folder)."""
    def work(job):
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, str(config.ROOT / "eval" / "run_eval.py"), "--sleep", str(gap)]
        if subject:
            cmd += ["--subject", subject]
        proc = subprocess.Popen(cmd,
                                cwd=str(config.ROOT), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line = line.rstrip()
            m = re.match(r"PROGRESS (\d+)/(\d+)", line)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                job["progress"] = i / n
            elif line and "Warning" not in line:
                job["steps"] = (job["steps"] + [line])[-200:]
        proc.wait()
        if proc.returncode:
            raise RuntimeError("evaluation stopped:\n" + "\n".join(job["steps"][-12:]))
        return "ok"
    return start("evaluate", work)