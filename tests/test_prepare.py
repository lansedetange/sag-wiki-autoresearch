from __future__ import annotations

import unittest
from types import SimpleNamespace

from prepare import PROJECT_ROOT, REFUSAL, evaluate_case, load_engine, retrieval_metrics


class FakeEngine:
    def __init__(self, retrieved_ids, answer_text, reported_ids) -> None:
        self.retrieved_ids = retrieved_ids
        self.answer_text = answer_text
        self.reported_ids = reported_ids

    def dual_layer_retrieve(self, query):
        hits = [
            (SimpleNamespace(chunk_id=raw_id), 1.0)
            for raw_id in self.retrieved_ids
        ]
        return [], hits

    def answer(self, query):
        return {
            "answer": self.answer_text,
            "cited_ids": self.reported_ids,
            "retrieved_raw_ids": self.retrieved_ids,
        }


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = {
            "id": "hotel",
            "question": "上海住宿费上限是多少？",
            "expected_raw_ids": ["policy_c00"],
            "required_facts": ["650"],
            "expect_refusal": False,
        }

    def test_expected_evidence_metrics(self) -> None:
        self.assertEqual(retrieval_metrics(["other_c00", "policy_c00"], ["policy_c00"]), (1.0, 0.5))
        self.assertEqual(retrieval_metrics(["other_c00"], ["policy_c00"]), (0.0, 0.0))

    def test_unanswerable_case_rewards_no_false_positive(self) -> None:
        self.assertEqual(retrieval_metrics([], []), (1.0, 1.0))
        self.assertEqual(retrieval_metrics(["other_c00"], []), (0.0, 0.0))

    def test_grounded_answer_receives_full_score(self) -> None:
        engine = FakeEngine(
            ["policy_c00"],
            "上海住宿费上限为650元 [policy_c00]。",
            ["policy_c00"],
        )
        row = evaluate_case(engine, self.case, "full")
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(row["citation_valid"], 1.0)
        self.assertEqual(row["answer_correct"], 1.0)

    def test_reported_citations_must_match_answer_body(self) -> None:
        engine = FakeEngine(
            ["policy_c00"],
            "上海住宿费上限为650元 [policy_c00]。",
            [],
        )
        row = evaluate_case(engine, self.case, "full")
        self.assertEqual(row["citation_valid"], 0.0)
        self.assertEqual(row["answer_correct"], 0.0)
        self.assertLess(row["score"], 1.0)

    def test_citation_must_be_retrieved(self) -> None:
        engine = FakeEngine(
            ["other_c00"],
            "上海住宿费上限为650元 [policy_c00]。",
            ["policy_c00"],
        )
        row = evaluate_case(engine, self.case, "full")
        self.assertEqual(row["citation_valid"], 0.0)
        self.assertEqual(row["answer_correct"], 0.0)

    def test_refusal_with_extra_claim_is_not_correct(self) -> None:
        case = {
            "id": "unknown",
            "question": "公司是否提供宠物补贴？",
            "expected_raw_ids": [],
            "required_facts": [],
            "expect_refusal": True,
        }
        engine = FakeEngine([], f"{REFUSAL}但可以提供999元。", [])
        row = evaluate_case(engine, case, "full")
        self.assertEqual(row["answer_correct"], 0.0)

    def test_retrieval_mode_does_not_claim_answer_metrics(self) -> None:
        engine = FakeEngine(["policy_c00"], "", [])
        row = evaluate_case(engine, self.case, "retrieval")
        self.assertEqual(row["score"], 1.0)
        self.assertIsNone(row["citation_valid"])
        self.assertIsNone(row["answer_correct"])

    def test_dynamic_engine_import_supports_dataclasses(self) -> None:
        engine_class = load_engine(PROJECT_ROOT / "sag_wiki_engine.py")
        self.assertTrue(hasattr(engine_class, "answer"))


if __name__ == "__main__":
    unittest.main()
