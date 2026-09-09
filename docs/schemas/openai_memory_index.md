# OpenAI-memory write-index dump (`openai_memory_index.v1`)

Extract-all dump used by the `openai_memory` MemoryBuilder.

**Code:** `src/locomo_eval/openai_memory/`  
**CLI:** `python -m src.locomo_eval.openai_memory.run_index`  
**Dump:** `experiments/<run_id>/openai_memory_index/`

The Mem0 paper's OpenAI row used ChatGPT's product Memory UI and then
injected **all** generated memories (privileged; no selective retrieve API).
This dump clones that protocol with a local extractor. It cannot reproduce
ChatGPT Memory and is **not** Table 2 J=52.90.

Gold answers never enter the dump.

| Path | Contents |
|------|----------|
| `run_meta.json` | extractor model, data SHA, `retrieve=all` |
| `schema.json` | `openai_memory_index.v1` |
| `index.jsonl` | one row per sample |
| `by_sample/<id>/transcript.txt` | source dialog |
| `by_sample/<id>/memories.json` | `{speaker, timestamp, text}` list |

QA concatenates every extracted entry. There is no top-k.
