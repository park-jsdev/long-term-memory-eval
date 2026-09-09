# Mem0 write-index dump (`mem0_index.v1`)

**Purpose:** inspectable extract+update (and optional graph) artifacts so a later eval command can retrieve without re-extracting.

**Code:** `src/locomo_eval/mem0/`  
**CLI:** `python -m src.locomo_eval.mem0.run_index`  
**Layout:** `experiments/<run_id>/mem0_index/`

This is an architecture clone of Mem0 / Mem0g ([arXiv:2504.19413](https://arxiv.org/abs/2504.19413)), not a reproduction of paper Table 1–2 J. No Neo4j, Qdrant, or Mem0 Platform client.

Gold answers never appear in these files.

---

## Folder

| Path | Contents |
|------|----------|
| `schema.json` | `mem0_index.v1` field map |
| `run_meta.json` | extract/update/embed models, `batch_size`, `similar_s`, `top_k`, `enable_graph`, data SHA, git hash |
| `index.jsonl` | one row per sample (fact counts, paths) |
| `by_sample/<sample_id>/speaker_a.json` | vector facts + embeddings for speaker A’s index |
| `by_sample/<sample_id>/speaker_b.json` | same for speaker B (role-flipped ingest) |
| `by_sample/<sample_id>/graph.json` | nodes + edges when `enable_graph` (`mem0g`) |
| `by_sample/<sample_id>/ingest_log.jsonl` | pair id, `turn_ids`, ops applied |

Complete sample (required before a later retrieve/format load): both speaker JSON files, plus `graph.json` if graph is on.

---

## Ingest protocol (eval `add.py`)

- Input unit is a HLD (i) `SessionBlock` (LoCoMo `session_N`, not re-cut).
- Consecutive turns in `batch_size=2` (remainder singleton).
- Two indexes per sample; role-flip so each speaker is `user` on their index.
- Content is `"Name: text"`; timestamp is `date_time_raw`.
- Extract runs on **user-role messages only**.

---

## Vector ops

`ADD | UPDATE | DELETE | NONE` against top `s=10` similar facts (cosine over `text-embedding-3-small` live, hashed bag-of-tokens mock).

---

## Graph (`mem0g`) — locked baseline store

Whenever a condition uses a graph (Mem0 write-index **or** teacher-built
memory), the store is **`Mem0GraphMemory`** (`src/locomo_eval/mem0/graph_memory.py`):

- `GraphMemory` ABC; `Mem0GraphMemory` is the only production implementation.
- Nodes: `node_id`, `name`, `entity_type`, `embedding`, `timestamp`.
- Edges: `source -- relationship -- target`, `valid`, `timestamp`, `edge_id`.
- Node reuse if cosine ≥ `t=0.7`. Conflicting edges get `valid=false`
  (paper invalidation, not Cypher DELETE).
- Teachers write through `ingest_triples` (already-extracted entities/relations)
  so fusion cannot invent a second schema.
- A later distilled graph is a new `GraphMemory` subclass; freeze extract when
  that is the claim.

---

## Read seam

`mem0` / `mem0g` MemoryBuilders load this dump, embed the **question**, retrieve `top_k=30` per speaker, and format `{timestamp}: {memory}` (+ `source -- relationship -- target` for mem0g). Missing dumps raise an error that names `run_index`.
