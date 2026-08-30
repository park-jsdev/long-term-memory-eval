# Phase 1: Minimal LoCoMo Baseline Pipeline

## Objective

Build a small, reproducible pipeline that runs a single LoCoMo question-answering baseline end to end.

The pipeline should:

1. load the released LoCoMo dataset,
2. construct the baseline context for each question,
3. send the context and question to a fixed QA model,
4. save the generated answer,
5. evaluate the generated answer against the LoCoMo reference answer.

Do not implement custom memory synthesis, multi-teacher generation, online memory updates, or new evaluation metrics in this phase.

## Scope

Implement only the LoCoMo question-answering task.

Support one initial baseline:

* session-summary memory, using the session summaries already included in the released dataset.

The design should make it straightforward to add other context sources later, such as:

* raw dialogue turns,
* generated observations,
* custom structured memories.

## Input

Use the LoCoMo JSON dataset.

Each conversation sample contains:

* `sample_id`
* conversation sessions and dialogue turns
* `session_summary`
* `observation`
* `qa`

Each QA item contains:

* question
* reference answer
* category
* evidence dialogue IDs, when available

The code should not modify the source dataset.

## Output

Produce one JSONL prediction file.

Each row should contain:

```json
{
  "sample_id": "conv-1",
  "question_id": "conv-1-q-12",
  "question": "What hobby did Caroline begin?",
  "reference_answer": "She began painting.",
  "predicted_answer": "Caroline began painting.",
  "category": "single-hop",
  "memory_type": "session_summaries",
  "memory_text": "...",
  "reader_model": "MODEL_NAME",
  "prompt_version": "v1"
}
```

Also produce one summary metrics file:

```json
{
  "number_of_questions": 100,
  "memory_type": "session_summaries",
  "reader_model": "MODEL_NAME",
  "metrics": {
    "exact_match": 0.0,
    "token_f1": 0.0
  }
}
```

## Pipeline

### 1. Dataset loader

Implement a loader that converts the LoCoMo JSON into normalized Python objects:

```python
Conversation
Question
Memory
Prediction
```

Assign a deterministic question ID if the dataset does not provide one.

### 2. Memory adapter

Define a common interface:

```python
class MemoryBuilder:
    def build(self, conversation: Conversation, question: Question) -> Memory:
        ...
```

Implement one version:

```python
SessionSummaryMemoryBuilder
```

For Phase 1, it may concatenate all available session summaries in chronological order.

Do not implement retrieval yet.

### 3. Prompt builder

Use one fixed prompt template:

```text
You are answering a question about a conversation.

Use only the supplied memory.
If the answer is not supported by the memory, say "Unknown."
Answer concisely.

Memory:
{memory}

Question:
{question}

Answer:
```

Store the prompt template in a separate file or named configuration.

### 4. Reader interface

Define a model-independent interface:

```python
class Reader:
    def answer(self, memory: str, question: str) -> str:
        ...
```

Implement one reader initially.

The reader model name, temperature, maximum output tokens, and API settings must be configurable.

Use deterministic settings where supported:

```text
temperature = 0
```

### 5. Evaluation

Implement:

* normalized exact match,
* token-level F1.

Normalize answers by:

* lowercasing,
* removing punctuation,
* removing articles,
* collapsing whitespace.

Report aggregate scores and scores by LoCoMo question category.

Do not add an LLM judge in Phase 1.

### 6. Command-line interface

The pipeline should run with one command:

```bash
python -m locomo_eval.run \
  --data data/locomo10.json \
  --memory session_summaries \
  --reader MODEL_NAME \
  --output outputs/baseline.jsonl
```

A separate evaluation command is acceptable:

```bash
python -m locomo_eval.offline_evaluate \
  --predictions outputs/baseline.jsonl
```

## Suggested repository structure

```text
locomo-memory-research/
├── README.md
├── pyproject.toml
├── configs/
│   └── mem0_baseline.yaml
├── data/
│   └── README.md
├── prompts/
│   ├── qa_v1.txt
│   └── qa_mem0_v1.txt
├── src/
│   └── locomo_eval/
│       ├── dataset.py
│       ├── schemas.py
│       ├── memory.py
│       ├── prompts.py
│       ├── readers.py
│       ├── metrics.py
│       ├── run.py
│       └── offline_evaluate.py
├── tests/
│   ├── test_dataset.py
│   ├── test_memory.py
│   └── test_metrics.py
└── outputs/
```

## Acceptance criteria

The implementation is complete when:

1. the LoCoMo dataset loads without manual editing,
2. one QA example can be run locally,
3. the full QA set can be run with the same command,
4. predictions are saved in deterministic JSONL format,
5. exact-match and token-F1 scores are produced,
6. model and prompt configuration are recorded in the output,
7. the memory builder can later be replaced without changing the reader or evaluator,
8. unit tests cover dataset parsing, memory construction, and answer normalization.

## Non-goals

Do not implement:

* multi-teacher synthesis,
* claim extraction,
* claim fusion,
* custom schemas,
* retrieval ranking,
* online memory updates,
* event-summarization evaluation,
* multimodal dialogue generation,
* training or fine-tuning,
* a web interface,
* distributed execution.

## Reproducibility requirements

Record:

* dataset version or commit hash,
* model name and version,
* prompt version,
* decoding parameters,
* timestamp,
* code commit hash.

Each pipeline invocation starts from scratch so interrupted runs do not mix with later results.
