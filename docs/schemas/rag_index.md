# RAG write-index dump (`rag_index.v1`)

Offline token-chunk + embed index used by the `rag` MemoryBuilder.

**Code:** `src/locomo_eval/rag/`  
**CLI:** `python -m src.locomo_eval.rag.run_index`  
**Dump:** `experiments/<run_id>/rag_index/`

This clones Mem0 `evaluation/src/rag.py` (tiktoken `cl100k_base` windows,
`text-embedding-3-small`, cosine top-k, join `\n<->\n`). It is **not** a
claim of paper Table 2 J.

Gold answers never enter the dump.

| Path | Contents |
|------|----------|
| `run_meta.json` | chunk_size, encoding, embedder, data SHA |
| `schema.json` | `rag_index.v1` |
| `index.jsonl` | one row per sample |
| `by_sample/<id>/transcript.txt` | timestamped dialog used for chunking |
| `by_sample/<id>/chunks.json` | `{chunk_id, text, n_tokens, embedding}` |

At QA time the builder embeds the **question** only and retrieves top-k
chunks. Full-context does **not** use this dump (`full_context` formats the
transcript live).
