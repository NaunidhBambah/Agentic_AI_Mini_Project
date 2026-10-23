# Model card: DoubtLoop

**System type:** LLM agent workflow (LangGraph) with retrieval, guardrails and human escalation.
**Version:** 1.0 (October 2026)

## Intended use
- Answering students' conceptual doubts about one course, using only that course's notes.
- Routing uncertain, administrative or uncovered doubts to a human TA.
- Showing faculty which topics confuse students.

## Out of scope
- Grading, marks or official decisions (always a human).
- Giving full solutions to graded assignments (hint-only by design).
- Medical, legal, financial or personal counselling.
- Questions unrelated to the course (refused).

## Components and data
- LLM: `openai/gpt-oss-120b` via Groq free tier (configurable; Gemini optional).
- Embeddings: `all-MiniLM-L6-v2` (ChromaDB default, runs locally).
- Knowledge: course notes in `data/course_notes/` (sample notes are included; replace with faculty material).
- Learned memory: answers written by TAs (stored with the TA's name and date).
- Seed data: 60 synthetic doubt records and 4 TA answers, flagged `is_seed=1`, for demonstration only.

## Evaluation
Run `python eval/run_eval.py` (40 labelled questions) and `python eval/learning_curve.py`.
Record your results here before submission:

| Metric | Result |
|---|---|
| Behaviour accuracy | |
| Answerable: task completion | |
| Escalation precision / recall | |
| Guardrail accuracy | |
| Escalation rate week 1 → week 3 | |

## Known limitations
- Answers are only as good as the notes; outdated notes give outdated answers.
- The LLM can still misjudge grounding; the critic reduces but does not remove hallucination.
- Regex guardrails can miss reworded attacks or unusual personal data formats.
- A wrong TA answer is also "learned"; there is no second review step yet.
- Free hosting may reset stored data after restarts.

## Ethical considerations and privacy
- Phone numbers, emails and long ID numbers are masked before storage.
- No data is used for model training by this project. Questions are sent to the LLM provider's API, so students should not paste private information.
- Humans stay accountable: TAs answer what the agent cannot, and every answer shows who produced it (notes, TA memory, or a named TA).
