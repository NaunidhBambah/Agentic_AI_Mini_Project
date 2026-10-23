"""PDF exports.

student_notes_pdf : a student's doubts with answers and sources -> personal revision notes
faculty_report_pdf: class summary (metrics, escalation trend chart, confusing topics, guardrails, insights)

Uses fpdf2 with the DejaVu fonts that ship with matplotlib, so symbols like → ≤ ✓ print correctly.
"""
import io
import os
import re
from collections import Counter
from datetime import datetime

import matplotlib
from fpdf import FPDF

from . import config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
INK, LILAC, HAZE = (35, 32, 80), (110, 84, 214), (110, 108, 150)
STATUS_TEXT = {
    "answered": "From course notes", "verified": "Verified by TA", "ta_answered": "Answered by TA",
    "hint": "Hint mode (graded work)", "escalated": "Waiting for a TA", "refused": "Outside this course",
    "blocked": "Blocked by guardrail", "history": "From your doubt history",
    "awaiting_student": "Not in the notes (not sent yet)", "withdrawn": "Not sent to a TA",
    "smalltalk": "DoubtLoop",
}


class _PDF(FPDF):
    def __init__(self, title: str):
        super().__init__(format="A4")
        self.doc_title = title
        self.add_font("DejaVu", "", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"))
        self.add_font("DejaVu", "I", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf"))
        self.set_auto_page_break(True, margin=16)
        self.set_margins(18, 16, 18)

    def header(self):
        self.set_font("DejaVu", "B", 9)
        self.set_text_color(*LILAC)
        self.cell(0, 6, "DoubtLoop", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*LILAC)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*HAZE)
        self.cell(0, 6, f"{self.doc_title}  |  page {self.page_no()}", align="C")

    def h1(self, text):
        self.set_font("DejaVu", "B", 18)
        self.set_text_color(*INK)
        self.multi_cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")

    def h2(self, text):
        self.ln(3)
        self.set_font("DejaVu", "B", 12.5)
        self.set_text_color(*INK)
        self.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")

    def small(self, text, color=HAZE):
        self.set_font("DejaVu", "", 8.5)
        self.set_text_color(*color)
        self.multi_cell(0, 4.6, text, new_x="LMARGIN", new_y="NEXT")

    def body(self, text):
        self.set_font("DejaVu", "", 10)
        self.set_text_color(30, 30, 40)
        self.multi_cell(0, 5.2, text, markdown=True, new_x="LMARGIN", new_y="NEXT")


def _clean(md: str) -> str:
    """Light markdown -> text that fpdf's markdown mode handles (keeps **bold**)."""
    md = md or ""
    md = re.sub(r"```.*?```", "[code omitted]", md, flags=re.S)
    md = re.sub(r"^#{1,6}\s*", "", md, flags=re.M)
    md = re.sub(r"^\s*[-*]\s+", "• ", md, flags=re.M)
    md = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+)\*(?!\*)", r"\1", md)   # single *italic* -> plain
    md = re.sub(r"_([^_\n]+)_", r"\1", md)
    return md.strip()


def student_notes_pdf(student: str, doubts: list, subject: str = None) -> bytes:
    pdf = _PDF(f"Doubt notes for {student}")
    pdf.add_page()
    pdf.h1(f"Doubt notes: {student}")
    pdf.small(f"{subject or config.COURSE_NAME}. Exported {datetime.now():%d %b %Y, %H:%M}. "
              f"{len(doubts)} doubts, oldest first.")
    pdf.ln(2)
    for i, d in enumerate(doubts, start=1):
        pdf.h2(f"{i}. {d['question']}")
        meta = STATUS_TEXT.get(d["status"], d["status"])
        if d.get("answered_by", "").startswith(("ta:", "memory:")) and d["answered_by"] != "memory:history":
            meta += f", by {d['answered_by'].split(':', 1)[1]}"
        pdf.small(f"{meta}  |  {d['created_at'][:16]}", LILAC)
        if d["status"] == "escalated":
            pdf.body("Waiting for a TA's answer.")
        else:
            pdf.body(_clean(d.get("answer")))
        if d.get("web_answer"):
            pdf.small("From the web (student-approved search, not from the course notes):", LILAC)
            pdf.body(_clean(d["web_answer"]))
            links = [s.get("title", "") for s in (d.get("web_sources") or [])]
            if links:
                pdf.small("Web sources: " + "; ".join(links))
        srcs = [s["label"] for s in (d.get("sources") or []) if s.get("label")]
        if srcs and d["status"] in ("answered", "hint", "verified"):
            pdf.small("Sources: " + "; ".join(srcs))
    return bytes(pdf.output())


def _trend_png(weekly) -> bytes:
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=160)
    ax.plot(list(weekly.index.strftime("%d %b")), list(weekly.values), marker="o", color="#6E54D6", lw=2.2)
    ax.set_ylabel("% escalated to TA")
    ax.set_ylim(0, max(60, float(weekly.max()) + 10))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def faculty_report_pdf(rows: list, metrics: dict, weekly, insights: str = "", subject: str = None) -> bytes:
    pdf = _PDF("Class doubt report")
    pdf.add_page()
    pdf.h1("Class doubt report")
    pdf.small(f"{subject or config.COURSE_NAME}. Generated {datetime.now():%d %b %Y, %H:%M}.")
    pdf.ln(3)

    pdf.h2("Summary")
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(30, 30, 40)
    with pdf.table(col_widths=(70, 40), text_align=("LEFT", "RIGHT"), first_row_as_headings=False) as t:
        for k, v in metrics.items():
            r = t.row()
            r.cell(k)
            r.cell(str(v))

    if weekly is not None and len(weekly):
        pdf.h2("Escalation rate by week")
        pdf.image(io.BytesIO(_trend_png(weekly)), w=170)
        pdf.small("Completed weeks with at least 5 doubts. A falling line means the system needs the TA less.")

    pdf.h2("Most confusing topics")
    topics = Counter(r["topic"] for r in rows if r["category"] == "concept").most_common(8)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(30, 30, 40)
    with pdf.table(col_widths=(110, 30), text_align=("LEFT", "RIGHT")) as t:
        h = t.row()
        h.cell("Topic")
        h.cell("Doubts")
        for name, n in topics:
            r = t.row()
            r.cell(name)
            r.cell(str(n))

    if insights:
        pdf.h2("Insights for the next lecture")
        pdf.body(_clean(insights))

    glog = [(r["created_at"][5:16], g["guardrail"].replace("_", " "), r["question"][:70])
            for r in rows for g in (r.get("guardrails") or [])]
    if glog:
        pdf.h2("Guardrail log")
        pdf.set_font("DejaVu", "", 8.5)
        pdf.set_text_color(30, 30, 40)
        with pdf.table(col_widths=(24, 38, 112)) as t:
            h = t.row()
            for c in ("Time", "Guardrail", "Question"):
                h.cell(c)
            for row in glog[:40]:
                r = t.row()
                for c in row:
                    r.cell(c)

    pending = [r for r in rows if r["status"] == "escalated"]
    if pending:
        pdf.h2(f"Still waiting for a TA ({len(pending)})")
        for r in pending[:20]:
            pdf.body(f"• {r['question']}")
    return bytes(pdf.output())