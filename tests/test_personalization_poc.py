"""Proof-of-concept personalization records, decisions, and counts.

No Codex binary and no API. The pack test runs the marker hook.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.locomo_plus import count_cognitive_items, load_cognitive_items
from src.locomo_eval.interaction_patterns.speaker_repeat_census import repeated_speaker_phrases
from src.locomo_eval.interaction_patterns.condition_decisions import (
    bounded_instruction,
    decide_request,
    policy_body,
    stage1_selection,
)
from src.locomo_eval.interaction_patterns.decision_pack_summary import summarize_pack
from src.locomo_eval.interaction_patterns.low_maintenance_example import LOW_MAINTENANCE
from src.locomo_eval.interaction_patterns.offline_decision_pack import build_personalization_poc_pack
from src.locomo_eval.interaction_patterns.hypothesis_rendering import h2_scores, truncate_to_token_budget
from src.locomo_eval.interaction_patterns.hypothesis_record import (
    HypothesisRecord,
    RequestLabel,
    validate_hypothesis,
    validate_request,
)


class TestValidateHypothesis(unittest.TestCase):
    def test_validate_hypothesis_rejects_inferred_record_when_evidence_ids_are_missing(self):
        record = HypothesisRecord(
            record_id="h",
            speaker="Maya",
            sample_id="s",
            who_basis="inferred",
            what_type="taste_or_value",
            what="Prefers quiet tools",
            scope="software tools",
            evidence_ids=[],
            session_ids=["1", "2"],
        )
        with self.assertRaises(ValueError):
            validate_hypothesis(record)

    def test_validate_hypothesis_rejects_inferred_record_when_only_one_session_is_cited(self):
        record = HypothesisRecord(
            record_id="h",
            speaker="Maya",
            sample_id="s",
            who_basis="inferred",
            what_type="taste_or_value",
            what="Prefers quiet tools",
            scope="software tools",
            evidence_ids=["D1:1"],
            session_ids=["1"],
        )
        with self.assertRaises(ValueError):
            validate_hypothesis(record)

    def test_validate_hypothesis_accepts_explicit_record_with_one_session(self):
        record = HypothesisRecord(
            record_id="h",
            speaker="Maya",
            sample_id="s",
            who_basis="explicit",
            what_type="stable_constraint",
            what="Needs a secure default",
            scope="software tools",
            evidence_ids=["D1:1"],
            session_ids=["1"],
        )
        validate_hypothesis(record)


class TestValidateRequest(unittest.TestCase):
    def test_validate_request_rejects_label_when_when_is_empty(self):
        label = RequestLabel(
            request_id="r",
            sample_id="s",
            speaker="Maya",
            request="Which tool?",
            if_decision="select",
            when="",
        )
        with self.assertRaises(ValueError):
            validate_request(label)


class TestTruncateToTokenBudget(unittest.TestCase):
    def test_truncate_to_token_budget_keeps_the_tail_inside_the_budget(self):
        text = "one two three four five six seven eight"
        truncated = truncate_to_token_budget(text, budget=8)
        self.assertLessEqual(len(truncated.split()), 8)
        self.assertTrue(truncated.endswith("seven eight"))


class TestH2Scores(unittest.TestCase):
    def test_h2_scores_counts_evidence_ids_only_when_they_appear_in_the_text(self):
        record = LOW_MAINTENANCE
        scores = h2_scores(
            [record],
            hypothesis_text="Evidence: D1:1 D3:2 Scope: software tools",
            summary_text="Maya likes software tools",
            unsupported_phrases=["buys a yacht"],
        )
        self.assertEqual(scores["hypothesis_evidence_coverage"], 1.0)
        self.assertEqual(scores["summary_evidence_coverage"], 0.0)
        self.assertEqual(scores["hypothesis_unsupported"], 0)
        self.assertEqual(scores["summary_unsupported"], 0)


class TestStage1Selection(unittest.TestCase):
    def test_stage1_selection_abstains_when_request_shares_no_scope_words(self):
        selected, abstain, reason = stage1_selection(
            [LOW_MAINTENANCE],
            "What is the capital of France?",
            condition="B",
            top_k=3,
        )
        self.assertEqual(selected, [])
        self.assertTrue(abstain)
        self.assertEqual(reason, "no_overlap")

    def test_stage1_selection_returns_the_record_for_condition_a_without_overlap(self):
        selected, abstain, reason = stage1_selection(
            [LOW_MAINTENANCE],
            "What is the capital of France?",
            condition="A",
            top_k=3,
        )
        self.assertEqual([record.record_id for record in selected], ["hyp-low-maint"])
        self.assertFalse(abstain)
        self.assertEqual(reason, "relevance_topk")


class TestDecideRequest(unittest.TestCase):
    def test_decide_request_reuses_the_same_selected_ids_for_b_and_c(self):
        decisions = decide_request(
            [LOW_MAINTENANCE],
            "Which software tool should I use for class notes?",
            "select-tool",
        )
        self.assertEqual(
            decisions["B"].selected_ids,
            decisions["C"].selected_ids,
        )
        self.assertEqual(decisions["C"].selected_ids, ["hyp-low-maint"])
        self.assertIn("low-maintenance", decisions["C"].stage2_instruction)

    def test_decide_request_omits_record_ids_from_reply_text_when_stages_are_model_output(self):
        decisions = decide_request(
            [LOW_MAINTENANCE],
            "What is the capital of France?",
            "abstain-fact",
            stage1_bc=([LOW_MAINTENANCE], False, "model"),
            stage2_text="Consider low-maintenance tools within software tools.",
            stage_prompts_used=True,
        )
        self.assertEqual(decisions["A"].reason, "relevance_topk")
        self.assertEqual(decisions["B"].selected_ids, ["hyp-low-maint"])
        self.assertEqual(decisions["C"].selected_ids, ["hyp-low-maint"])
        self.assertNotIn("hyp-low-maint", decisions["B"].text)
        self.assertNotIn("hyp-low-maint", decisions["P"].text)
        self.assertTrue(decisions["C"].stage_prompts_used)

    def test_decide_request_keeps_b_and_c_ids_when_stage2_text_is_empty(self):
        decisions = decide_request(
            [LOW_MAINTENANCE],
            "Which software tool should I use?",
            "select-tool",
            stage1_bc=([LOW_MAINTENANCE], False, "model"),
            stage2_text="",
            stage_prompts_used=True,
        )
        self.assertEqual(decisions["B"].selected_ids, ["hyp-low-maint"])
        self.assertEqual(decisions["C"].selected_ids, ["hyp-low-maint"])
        self.assertEqual(decisions["C"].action, "abstain")
        self.assertEqual(decisions["B"].action, "inject")

    def test_decide_request_writes_no_hook_policy_for_the_prompt_only_condition(self):
        decisions = decide_request(
            [LOW_MAINTENANCE],
            "Which software tool should I use?",
            "select-tool",
        )
        self.assertIsNone(policy_body(decisions["P"]))
        self.assertFalse(decisions["P"].hooks_enabled)


class TestBoundedInstruction(unittest.TestCase):
    def test_bounded_instruction_names_the_exception_the_request_contains(self):
        line = bounded_instruction(
            LOW_MAINTENANCE,
            ["explicit security requirement"],
        )
        self.assertIn("Do not override: explicit security requirement.", line)
        self.assertNotIn("hyp-low-maint", line)


class TestSummarizePack(unittest.TestCase):
    def test_summarize_pack_raises_when_b_and_c_selected_ids_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp)
            for condition, ids in (("B", ["hyp-low-maint"]), ("C", [])):
                self._write(pack, condition, ids)
            for condition in ("P", "A"):
                self._write(pack, condition, ["hyp-low-maint"])
            with self.assertRaises(ValueError):
                summarize_pack(pack, ["hyp-low-maint"])

    def _write(self, pack: Path, condition: str, ids: list[str]) -> None:
        folder = pack / "requests" / "sample" / condition / "memory"
        folder.mkdir(parents=True)
        decision = {
            "condition": condition,
            "action": "inject" if ids else "abstain",
            "selected_ids": ids,
            "abstain": not ids,
            "stage2_instruction": "",
        }
        (folder / "decision.json").write_text(json.dumps(decision), encoding="utf-8")


class TestRepeatedSpeakerPhrases(unittest.TestCase):
    def test_repeated_speaker_phrases_lists_a_word_that_appears_in_two_sessions(self):
        samples = [
            {
                "sample_id": "c1",
                "conversation": {
                    "session_1": [{"speaker": "Maya", "text": "I tend the garden daily."}],
                    "session_2": [{"speaker": "Maya", "text": "The garden needed water."}],
                    "session_3": [{"speaker": "Jon", "text": "I baked bread once."}],
                },
            }
        ]
        rows = repeated_speaker_phrases(samples, limit_per_speaker=10)
        maya = [row for row in rows if row["speaker"] == "Maya" and row["phrase"] == "garden"]
        self.assertEqual(len(maya), 1)
        self.assertEqual(maya[0]["session_ids"], ["1", "2"])
        self.assertFalse(any(row["speaker"] == "Jon" and row["phrase"] == "baked" for row in rows))


class TestLoadCognitiveItems(unittest.TestCase):
    def test_load_cognitive_items_reads_relation_type_and_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "locomo_plus.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "relation_type": "goal",
                            "cue_dialogue": "A: I am saving for a trip.\nB: Noted.",
                            "trigger_query": "A: What should I do with the bonus?",
                            "time_gap": "two weeks",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            items = load_cognitive_items(path)
        self.assertEqual(items[0].relation_type, "goal")
        self.assertTrue(items[0].cue_is_insertion)


class TestCountCognitiveItems(unittest.TestCase):
    def test_count_cognitive_items_marks_a_bare_cue_as_one_interaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "relation_type": "value",
                            "cue_dialogue": "A: I keep weekends free.",
                            "trigger_query": "A: Can we meet Saturday?",
                            "time_gap": "one week",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            report = count_cognitive_items(load_cognitive_items(path))
        self.assertEqual(report["n_items"], 1)
        self.assertEqual(report["by_relation"]["value"], 1)
        self.assertEqual(report["two_session_support"], 0)
        self.assertEqual(report["explicit_single_interaction"], 1)

    def test_count_cognitive_items_counts_two_sessions_when_evidence_prefixes_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "relation_type": "state",
                            "cue_dialogue": "A: I moved.",
                            "trigger_query": "A: Where should I store this?",
                            "time_gap": "a month",
                            "evidence": ["D1:1", "D4:2"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            report = count_cognitive_items(load_cognitive_items(path))
        self.assertEqual(report["two_session_support"], 1)
        self.assertEqual(report["cue_is_insertion"], 0)


class TestHypothesisFromCognitiveItem(unittest.TestCase):
    def test_hypothesis_from_cognitive_item_accepts_a_single_inserted_cue(self):
        from src.data.locomo_plus import CognitiveItem
        from src.locomo_eval.interaction_patterns.plus_item_to_record import hypothesis_from_cognitive_item

        record = hypothesis_from_cognitive_item(
            CognitiveItem(
                index=0,
                relation_type="goal",
                cue_dialogue="A: I am saving for a trip.",
                trigger_query="A: What should I do with the bonus?",
                time_gap="two weeks",
                evidence_ids=[],
                cue_is_insertion=True,
            )
        )
        self.assertEqual(record.who_basis, "explicit")
        self.assertEqual(record.evidence_ids, ["cue:0"])
        self.assertEqual(record.session_ids, ["cue"])
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dest = Path(cls._tmp.name)
        cls.report = build_personalization_poc_pack(cls.dest)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_build_personalization_poc_pack_keeps_b_and_c_selection_aligned(self):
        self.assertTrue(self.report["bc_identity_ok"])
        self.assertIsNone(self.report["context_used_appropriately"])

    def test_build_personalization_poc_pack_delivers_stage2_text_through_the_hook(self):
        folder = self.dest / "requests" / "select-tool" / "C"
        body = json.loads((folder / "hook_stdout.json").read_text(encoding="utf-8"))
        text = body["body"]["hookSpecificOutput"]["additionalContext"]
        self.assertIn("low-maintenance", text)
        self.assertNotIn("hyp-low-maint", text)
        log = json.loads((folder / "hook_log.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(log["action"], "inject")
        self.assertEqual(log["selected_ids"], ["hyp-low-maint"])

    def test_build_personalization_poc_pack_abstains_when_the_request_has_no_overlap(self):
        folder = self.dest / "requests" / "abstain-fact" / "B"
        body = json.loads((folder / "hook_stdout.json").read_text(encoding="utf-8"))
        self.assertEqual(body["body"], {})
        decision = json.loads((folder / "memory" / "decision.json").read_text(encoding="utf-8"))
        self.assertTrue(decision["abstain"])
        prompt_only = self.dest / "requests" / "abstain-fact" / "P"
        self.assertFalse((prompt_only / "memory" / "policy.json").exists())
        self.assertIn("Memories:", (prompt_only / "prompt.txt").read_text(encoding="utf-8"))

    def test_build_personalization_poc_pack_names_the_security_exception_in_stage2(self):
        decision = json.loads(
            (
                self.dest / "requests" / "override-security" / "C" / "memory" / "decision.json"
            ).read_text(encoding="utf-8")
        )
        self.assertIn("explicit security requirement", decision["stage2_instruction"])
        self.assertEqual(decision["selected_ids"], ["hyp-low-maint"])
        self.assertEqual(
            (self.dest / "requests" / "override-security" / "C" / "response.txt").read_text(
                encoding="utf-8"
            ),
            "DRY_RUN\n",
        )


class TestPlusOverlapSummary(unittest.TestCase):
    def test_plus_overlap_summary_counts_a_trigger_that_shares_no_content_words(self):
        from src.data.locomo_plus import CognitiveItem
        from src.locomo_eval.interaction_patterns.plus_starter_and_overlap import plus_overlap_summary

        items = [
            CognitiveItem(
                index=0,
                relation_type="causal",
                cue_dialogue="A: I learned to decline extra work.",
                trigger_query="A: I volunteered and now I am overwhelmed.",
                time_gap="two weeks later",
                evidence_ids=[],
                cue_is_insertion=True,
            )
        ]
        summary = plus_overlap_summary(items)
        self.assertEqual(summary["n_zero_overlap"], 1)
        self.assertEqual(summary["fraction_zero_overlap"], 1.0)


class TestLocomoQaOverlapSummary(unittest.TestCase):
    def test_locomo_qa_overlap_summary_counts_a_question_that_repeats_evidence_words(self):
        from src.locomo_eval.interaction_patterns.plus_starter_and_overlap import locomo_qa_overlap_summary

        samples = [
            {
                "sample_id": "c1",
                "conversation": {
                    "session_1": [
                        {"dia_id": "D1:1", "speaker": "Maya", "text": "I adopted a greyhound."}
                    ]
                },
                "qa": [
                    {
                        "question": "Did Maya mention a greyhound?",
                        "category": 4,
                        "evidence": ["D1:1"],
                    }
                ],
            }
        ]
        summary = locomo_qa_overlap_summary(samples)
        self.assertEqual(summary["n_questions"], 1)
        self.assertEqual(summary["fraction_question_overlaps_evidence"], 1.0)


class TestMergeSheet(unittest.TestCase):
    def test_merge_sheet_keeps_the_cue_and_a_human_revision_over_the_draft(self):
        from src.locomo_eval.interaction_patterns.plus_label_sheet import merge_sheet

        starter = [
            {
                "record_id": "plus-0",
                "what": "A: I learned to say no.",
                "request": "I am overwhelmed.",
                "relation_type": "causal",
            }
        ]
        drafts = {
            "plus-0": {
                "who_basis": "explicit",
                "what_type": "stable_constraint",
                "what": "Saying no reduced their stress.",
                "scope": "Extra commitments",
                "exceptions": [],
                "if_decision": "select",
                "when": "The trigger is overwhelm.",
                "how_then": "Do not tell them to quit.",
                "uncertainty": "One cue.",
                "confidence": "high",
            }
        }
        saved = [
            {
                "record_id": "plus-0",
                "label_status": "confirmed",
                "gold": True,
                "what": "Protecting time matters.",
                "scope": "Volunteering",
                "if_decision": "select",
                "when": "Edited.",
                "how_then": "Edited instruction.",
                "who_basis": "explicit",
                "what_type": "stable_constraint",
                "exceptions": [],
                "uncertainty": "Checked.",
                "annotator_notes": "ok",
            }
        ]
        merged = merge_sheet(starter, saved, drafts)
        self.assertEqual(merged[0]["cue_dialogue"], "A: I learned to say no.")
        self.assertEqual(merged[0]["what"], "Protecting time matters.")
        self.assertTrue(merged[0]["gold"])
        self.assertEqual(merged[0]["ai_what"], "Saying no reduced their stress.")

    def test_update_row_marks_gold_only_when_the_save_asks_for_it(self):
        from src.locomo_eval.interaction_patterns.plus_label_sheet import update_row

        rows = [{"record_id": "plus-1", "what": "old", "exceptions": []}]
        saved = update_row(rows, "plus-1", {"what": "new", "exceptions": "safety, cost", "gold": False})
        self.assertEqual(saved["label_status"], "revised")
        self.assertFalse(saved["gold"])
        self.assertEqual(saved["exceptions"], ["safety", "cost"])


class TestSmokePolicy(unittest.TestCase):
    def test_smoke_policy_keeps_the_record_id_out_of_injected_text(self):
        from harness.codex.spikes.run_personalization_smoke import cases

        select = next(case for case in cases() if case["cell_id"] == "c_select")
        self.assertIn("HARNESS_MARKER_ALPHA", select["policy"]["text"])
        self.assertNotIn("hyp-low-maint", select["policy"]["text"])
        self.assertEqual(select["policy"]["selected_ids"], ["hyp-low-maint"])
        self.assertNotIn("HARNESS_MARKER_ALPHA", select["prompt"])
        self.assertNotIn("hyp-low-maint", select["instruction"])


class TestPlusCodexSmokeCases(unittest.TestCase):
    def test_cases_from_pack_injects_the_saved_instruction_and_not_the_record_id(self):
        from harness.codex.spikes.run_plus_codex_smoke import cases_from_pack

        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp)
            self._write(pack, "plus-0", "C", "Say no to extra commitments.", ["plus-0"], False)
            self._write(pack, "plus-0", "B", "What: Saying no reduced stress.", ["plus-0"], False)
            self._write(pack, "plus-188", "C", "", [], True)
            loaded = {case["cell_id"]: case for case in cases_from_pack(pack)}
        self.assertIn("HARNESS_MARKER_ALPHA", loaded["plus0_c"]["policy"]["text"])
        self.assertNotIn("plus-0", loaded["plus0_c"]["policy"]["text"])
        self.assertEqual(loaded["plus0_c"]["policy"]["selected_ids"], ["plus-0"])
        self.assertEqual(loaded["plus188_c"]["policy"]["action"], "abstain")
        self.assertNotIn("HARNESS_MARKER_ALPHA", loaded["plus0_c"]["prompt"])
        self.assertIn("totally overwhelmed", loaded["plus0_c"]["prompt"])

    def _write(
        self,
        pack: Path,
        request_id: str,
        condition: str,
        text: str,
        selected_ids: list[str],
        abstain: bool,
    ) -> None:
        folder = pack / "requests" / request_id
        (folder / "N").mkdir(parents=True, exist_ok=True)
        (folder / "N" / "prompt.txt").write_text(
            "Request:\nI ended up volunteering and now I am totally overwhelmed.\n",
            encoding="utf-8",
        )
        memory = folder / condition / "memory"
        memory.mkdir(parents=True)
        decision = {
            "condition": condition,
            "text": text,
            "abstain": abstain,
            "selected_ids": selected_ids,
            "action": "abstain" if abstain else "inject",
        }
        (memory / "decision.json").write_text(json.dumps(decision), encoding="utf-8")
