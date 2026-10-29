# Reader (answer LLM)

Frozen sandwich bottom: `{memory}` + question → predicted answer.
Gold answers are in `predictions.jsonl` for scoring only; they are
not sent to the reader.

- `traces.jsonl` — reasoning / usage per question
- `predictions.jsonl` — LoCoMo QA rows (also copied at run root)
