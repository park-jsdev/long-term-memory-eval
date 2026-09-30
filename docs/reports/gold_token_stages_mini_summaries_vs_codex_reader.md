# Gold-token stages: mini summaries vs Codex reader

Same 1,540 judged LoCoMo questions. Category 5 is excluded, matching the Mem0 judge. A question counts only when every normalized gold token is present. Normalization is the same function as token F1 (`normalize_answer` in `src/locomo_eval/metrics.py`). Answer length is not in the denominator.

The conversation column is the stuffed transcript from the full-context cell `locomo-openai-codex-poc-readers-2675e258`. Both other cells are joined to it on `question_id`.

| | Every gold token in the conversation | Every gold token in the text the answer model read | Every gold token in the answer | Conversation had them, the middle text dropped some | Middle text had them, the answer dropped some |
|---|---:|---:|---:|---:|---:|
| Mini summaries writer | 0.777 (1,197) | 0.581 (895 summaries) | 0.342 (526) | 0.226 (348) | 0.276 (425) |
| Codex reader, persist off | 0.777 (1,197) | 0.620 (955 shell) | 0.321 (495) | 0.166 (255) | 0.324 (499) |

## What each column is

**Conversation.** The stuffed transcript, before anyone writes a summary or runs a search. It is 0.777 on both rows because both rows use the same full-context `memory_text`, joined on `question_id`. It is below 1.0 because many gold answers are not the words in the dialogue. "4 years" does not match "four years." "The sunday before 25 May 2023" does not match a line that only names the day of the race. `normalize_answer` lowercases and strips articles. It does not paraphrase. 1,197 is how often the gold wording is literally in the conversation. The other 343 judged questions never had those exact tokens to preserve. This column is not the model writing the summary.

**Text the answer model read.** For the mini row, this is the finished summary (`memory_text`). The reader does not search it. The whole summary is in the prompt. 0.581 means the gold tokens are still in that summary. For the Codex row, this is the completed shell output, not a summary.

**Answer.** The final scored string. For the mini row that is the Chat Completions reader. For the Codex row that is the agent's last message.

**Conversation had them, the middle text dropped some.** The writing loss on the mini row (348), and the search loss on the Codex row (255). The words were in the conversation and missing from the summary or the shell output.

**Middle text had them, the answer dropped some.** The reading loss. The words were in the summary (425) or the shell output (499) and missing from the answer.

For the summary row, fact finding is the finished summary, not the reader's answer and not the model's state while it was writing. Both stages are in the table. They are different columns.

## What we cannot see while the summary is written

The mini summary write is one Chat Completions call. The provider returns the summary. The pack stores that text and the token counts. `reasoning_tokens` is 0. There is no shell log and no thought log between the session text and the summary, so the 348-question drop is only the difference between those two strings.

A later proxy is whatever the API returns before the summary: a reasoning summary, or a chain-of-thought the prompt asks the model to emit and we save. That is still model text, not the provider's hidden state. `gpt-4o-mini` did not return a reasoning channel in this pack, so that proxy needs a model or a prompt that emits one.

Codex-as-writer is the exception. That loop runs in our process, the same `codex exec` as the reader. Those writer packs kept `output_text` and dropped the shell events, so the same staged check was possible and was not saved.

## What the two rows are

**Mini summaries writer.** Chat Completions `gpt-4o-mini` writes session summaries once. A frozen `gpt-4o-mini` reader with `prompts/readers/qa_mem0_v1.txt` then answers from that summary alone. Pack `locomo-openai-mini-writers-structured-c11a4c17`. The middle text is `memory_text` in `predictions.jsonl`.

**Codex reader, persist off.** The same model runs inside Codex. One `codex exec` per question, read-only sandbox, no notes carried across questions. The middle text is the completed shell output (`command_execution` in `agent/events.jsonl`), not a saved summary. Pack `locomo-openai-codex-poc-readers-54877192`. The answer prompt is `prompts/agents/qa_workspace_v1.txt`. The scored string is the last agent message.

## Result

The shell still holds every gold token on 955 questions. The summary holds them on 895. Where the conversation already had the full wording, the shell drops some on 255 questions and the summary drops some on 348.

The final answer goes the other way. The summary reader keeps every gold token on 526 questions. The Codex answer keeps them on 495. Given that the middle text was already complete, the answer drops a gold token on 0.475 of mini questions (425 / 895) and 0.523 of Codex questions (499 / 955).

## Two claims

**Fact finding.** On LoCoMo, Codex question-time search puts every gold token in view more often than a session summary written by `gpt-4o-mini`, and less often than the full conversation. The counts are 955 (shell), 895 (summary), and 1,197 (conversation). Where the conversation already had the full wording, the shell drops some on 255 questions and the summary drops some on 348.

**Scored answer.** That retrieval gain does not appear in the scored answer. The summary reader keeps every gold token on 526 questions. The Codex answer keeps them on 495. Given a complete middle text, the answer drops a gold token on 425 / 895 summary questions (0.475) and 499 / 955 Codex questions (0.523).

The answer columns are not two outputs of one reader. The summary row is a Chat Completions reader (`qa_mem0_v1`) answering from the summary. The Codex row is the agent answering from its own shell search (`qa_workspace_v1`). The middle column is the fair fact-finding comparison. The answer column mixes the memory with who writes the answer.

## What to run next

Freeze the reader. Give `qa_mem0_v1` the summary on one cell and the shell text on the other. If the shell-fed reader wins, the retrieval gain survives once the answer step is held fixed. If it does not, the drop is in that shared answer step, or in the shell text being a long transcript rather than a summary.

## Future work: answer-step thought logs

The summary contains every gold token on 895 of 1,540 judged questions. On 425 of those, the reader's answer drops at least one. The reader keeps every gold token on 526 questions. The words were in the summary, and the answer left some of them out. The same shape appears on the Codex row: the shell has every gold token on 955 questions, and the answer drops at least one on 499 of those.

The missing piece for comparing those drops is the answer step's thought log on both sides.

The summary reader has `reader/traces.jsonl` on `locomo-openai-mini-writers-structured-c11a4c17`: 1,986 rows, role `reader`, model `gpt-4o-mini`, the predicted answer, and token usage. `reasoning` is empty on every row, and `reasoning_tokens` is 0. The trace is the request result, not a thought between the summary and the answer.

Codex end-to-end has the shell log, which is the retrieval stage, and it has `agent_message` text in `agent/events.jsonl`. It does not have a separate reasoning channel. `reasoning_output_tokens` is 0 on every turn of `locomo-openai-codex-poc-readers-54877192`. An earlier agent message held every gold token while the final message lost it on 25 questions, so the saved utterances are mostly the same text as the scored answer.

A matched thought log has to be collected on both answer steps: the summary reader and the Codex answer. A reasoning summary, or a chain-of-thought the prompt asks the model to emit and we save, is still model text, not the provider's hidden state. `gpt-4o-mini` did not return a reasoning channel in these packs, so that proxy needs a model or a prompt that emits one. Codex-as-writer is a separate save: those packs kept `output_text` and dropped the shell events from the write.

## What these claims cannot say

- That Codex beats full context at finding facts. The conversation has every gold token on 1,197 questions. The shell has them on 955.
- That Codex is a better memory system overall. Stuffed-context J on this model is 0.744. Persist-off Codex J is 0.673.
- That the 2024 model throws facts away only inside the agent. The summary reader drops a complete gold phrase on 425 questions with no agent loop.
- That prompt tokens or shell byte counts measure this. This table measures whether the gold words are present.
