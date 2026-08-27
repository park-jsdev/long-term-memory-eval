# Trace: 2026-08-27 — before Mem0-parity reader default

Default reader was `gpt-4.1-mini` + `prompts/qa_v1.txt`. Writer for mem0
extract was already `gpt-4o-mini`. Live files now default the reader to
`gpt-4o-mini` + `prompts/qa_mem0_v1.txt` (Mem0 ANSWER_PROMPT adapted to
`{memory}`/`{question}`). `qa_v1.txt` and `gpt-4.1-mini` remain overrides.
