# Third-party terms

The MIT license in `LICENSE` covers the code in this repository,
Long-Term Memory Eval (Copyright (c) 2026 Junsoo Park and Bryan Triana). It does not
cover the dataset or the third-party prompts described below, which carry their
own terms.

## LoCoMo dataset — CC BY-NC 4.0

`data/raw/locomo10.json` is **not** distributed here. `scripts/fetch_locomo.py`
downloads it from the upstream release, and `data/raw/` is gitignored.

The dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
so **commercial use is not permitted**. Non-commercial research use must
attribute the original authors.

> Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y.
> (2024). *Evaluating Very Long-Term Conversational Memory of LLM Agents.*
> arXiv:2402.17753.

Pinned upstream commit: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`.

## Mem0 prompts — Apache-2.0

Several files under `prompts/` are transcribed from the Mem0 open-source
repository, which is licensed under Apache-2.0. Each file records the upstream
commit it was pinned at:

| Prompt | Upstream pin |
|---|---|
| `prompts/readers/qa_mem0_v1.txt` | released Mem0 answer prompt |
| `prompts/autoraters/autorater_mem0_v1.txt` | `ACCURACY_PROMPT`, mem0 @ `ece7ff6b` |
| `prompts/writers/mem0_extract_v1.txt`, `mem0_update_v1.txt` | mem0 @ `ece7ff6b` |
| `prompts/writers/mem0g_*.txt` | mem0 graph @ `69a832dc` |

> Chhikara, P., Khant, D., Aryan, S., Singh, T., & Yadav, D. (2025).
> *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.*
> arXiv:2504.19413.

The Mem0 baseline values in `src/locomo_eval/mem0_baselines.py` are literature
pins quoted from that paper for comparison. They are **not** reproductions, and
numbers produced by this repository's clones must not be reported as the
paper's.

## Papers

Cited PDFs are not redistributed here. `papers/` is gitignored; fetch papers
from arXiv or the publisher.

## Model providers

Live runs call the OpenAI, Anthropic, and DeepSeek APIs under each provider's
own terms of service. You supply your own keys; none are included.
