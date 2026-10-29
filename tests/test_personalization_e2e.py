"""Model stages and the first-pass sheet pack. No API key.

The pack test runs the marker hook. The completer is a fake, so a
failure here is the decision wiring, not a model answer.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.interaction_patterns.labeled_plus_run import (
    build_first_pass_e2e_pack,
    score_harness_decision,
)
from src.locomo_eval.interaction_patterns.hypothesis_rendering import (
    render_memory_for_reply,
    render_record_for_decision,
)
from src.locomo_eval.interaction_patterns.hypothesis_record import HypothesisRecord
from src.locomo_eval.interaction_patterns.labeled_row_to_record import (
    gold_request_from_labeled_row,
    memory_record_from_labeled_row,
)
from src.locomo_eval.interaction_patterns.harness_stage_calls import (
    parse_stage1_response,
    strip_record_ids,
)

LABELED_ROW = {
    "record_id": "plus-0",
    "request": "I ended up volunteering and now I am overwhelmed.",
    "who_basis": "explicit",
    "what_type": "stable_constraint",
    "what": "Saying no has reduced their stress.",
    "scope": "Taking on extra commitments",
    "exceptions": [],
    "uncertainty": "One inserted cue.",
    "if_decision": "select",
    "when": "GOLD WHEN volunteering overwhelm applies.",
    "how_then": "GOLD THEN acknowledge overload without ordering them to quit.",
}


class ScriptedCompleter:
    """Returns one Stage 1 selection and one instruction. Replies are fixed."""

    def complete(self, messages, *, json_mode=False):
        blob = "\n".join(message.get("content") or "" for message in messages)
        if "GOLD WHEN" in blob or "GOLD THEN" in blob:
            raise AssertionError("gold decision text reached the completer")
        system = messages[0]["content"] if messages else ""
        if "instruction stage" in system:
            return "Acknowledge the overload. Stay within extra commitments."
        if json_mode or "decision stage" in system:
            return json.dumps(
                {
                    "selected_ids": ["plus-0"],
                    "abstain": False,
                    "reason": "the request is about extra commitments",
                }
            )
        return "A short reply."


class TestMemoryRecordFromLabeledRow(unittest.TestCase):
    def test_memory_record_from_labeled_row_keeps_scope_and_drops_the_gold_decision(self):
        record = memory_record_from_labeled_row(LABELED_ROW)
        self.assertEqual(record.scope, "Taking on extra commitments")
        self.assertEqual(record.evidence_ids, ["cue:0"])
        blob = json.dumps(record.to_dict())
        self.assertNotIn("GOLD WHEN", blob)
        self.assertNotIn("GOLD THEN", blob)


class TestGoldRequestFromLabeledRow(unittest.TestCase):
    def test_gold_request_from_labeled_row_marks_the_record_applicable_on_select(self):
        label = gold_request_from_labeled_row(LABELED_ROW)
        self.assertEqual(label.if_decision, "select")
        self.assertEqual(label.applicable_ids, ["plus-0"])
        self.assertIn("GOLD THEN", label.how_then)


class TestParseStage1Response(unittest.TestCase):
    def test_parse_stage1_response_abstains_when_the_body_is_not_json(self):
        ids, abstain, reason = parse_stage1_response("not json", ["plus-0"])
        self.assertEqual(ids, [])
        self.assertTrue(abstain)
        self.assertEqual(reason, "parse_error")

    def test_parse_stage1_response_drops_an_unknown_id_and_abstains(self):
        body = json.dumps(
            {"selected_ids": ["other"], "abstain": False, "reason": "guess"}
        )
        ids, abstain, reason = parse_stage1_response(body, ["plus-0"])
        self.assertEqual(ids, [])
        self.assertTrue(abstain)
        self.assertEqual(reason, "guess")


class TestStripRecordIds(unittest.TestCase):
    def test_strip_record_ids_removes_the_longer_id_before_its_prefix(self):
        cleaned = strip_record_ids("see plus-101 and plus-1", ["plus-1", "plus-101"])
        self.assertNotIn("plus-101", cleaned)
        self.assertNotIn("plus-1", cleaned)


class TestRenderMemoryForReply(unittest.TestCase):
    def test_render_memory_for_reply_omits_the_record_id(self):
        record = HypothesisRecord(
            record_id="plus-0",
            speaker="A",
            sample_id="plus-0",
            who_basis="explicit",
            what_type="stable_constraint",
            what="Saying no has reduced their stress.",
            scope="Taking on extra commitments",
            evidence_ids=["cue:0"],
            session_ids=["cue"],
        )
        text = render_memory_for_reply(record)
        self.assertIn("Saying no has reduced their stress.", text)
        self.assertIn("this user", text)
        self.assertNotIn("speaker", text)
        self.assertNotIn("plus-0", text)
        self.assertNotIn("cue:0", text)


class TestRenderRecordForDecision(unittest.TestCase):
    def test_render_record_for_decision_names_this_user_and_keeps_the_record_id(self):
        record = memory_record_from_labeled_row(LABELED_ROW)
        text = render_record_for_decision(record)
        self.assertIn("this user", text)
        self.assertIn("[plus-0]", text)
        self.assertNotIn("speaker", text)


class TestScoreHarnessDecision(unittest.TestCase):
    def test_score_harness_decision_matches_when_stage1_selects_the_gold_id(self):
        from src.locomo_eval.interaction_patterns.condition_decisions import decide_request

        record = memory_record_from_labeled_row(LABELED_ROW)
        gold = gold_request_from_labeled_row(LABELED_ROW)
        decisions = decide_request(
            [record],
            gold.request,
            gold.request_id,
            stage1_bc=([record], False, "model"),
            stage2_text="Acknowledge the overload.",
            stage_prompts_used=True,
        )
        score = score_harness_decision(gold, decisions)
        self.assertTrue(score["if_match"])
        self.assertEqual(score["record_id_in_instruction"], [])


class TestBuildFirstPassE2ePack(unittest.TestCase):
    def test_build_first_pass_e2e_pack_keeps_gold_decision_text_out_of_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            summary = build_first_pass_e2e_pack(
                dest,
                [LABELED_ROW],
                ScriptedCompleter(),
                answer=False,
            )
            item = summary["items"][0]
            self.assertTrue(item["if_match"])
            self.assertEqual(item["gold_fields_in_prompts"], [])
            self.assertEqual(item["predicted_if"], "select")
            cue = "Saying no has reduced their stress."
            folder = dest / "requests" / "plus-0"
            bare = (folder / "N" / "prompt.txt").read_text(encoding="utf-8")
            prompt_c = (folder / "C" / "prompt.txt").read_text(encoding="utf-8")
            prompt_b = (folder / "B" / "prompt.txt").read_text(encoding="utf-8")
            self.assertNotIn(cue, bare)
            self.assertNotIn("plus-0", prompt_b)
            self.assertIn(cue, prompt_b)
            self.assertIn("Acknowledge the overload", prompt_c)
            self.assertNotIn("GOLD THEN", prompt_c)

    def test_build_first_pass_e2e_pack_delivers_the_instruction_without_the_record_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            build_first_pass_e2e_pack(
                dest,
                [LABELED_ROW],
                ScriptedCompleter(),
                answer=False,
            )
            folder = dest / "requests" / "plus-0"
            hook = json.loads((folder / "C" / "hook_stdout.json").read_text(encoding="utf-8"))
            delivered = hook["body"]["hookSpecificOutput"]["additionalContext"]
            self.assertIn("Acknowledge the overload", delivered)
            self.assertNotIn("plus-0", delivered)
            response = (folder / "N" / "response.txt").read_text(encoding="utf-8")
            self.assertEqual(response, "DRY_RUN\n")
