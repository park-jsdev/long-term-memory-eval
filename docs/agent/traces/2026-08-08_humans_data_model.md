# Trace: 2026-08-08 — HUMANS data/task mental model

Added section **Mental model of the data & task** to `docs/agent/HUMANS.md`:

- LoCoMo `locomo10.json` = official dialog + aids + gold QA
- `qa_all` = our flatten export for inspection
- Answer LLM never sees gold; scorer does
- Memory design = how we build Memory.text accompanying the question
