from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sag_wiki_engine import CONFIG, REFUSAL, SAGWikiEngine


class FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        content = self.responses.pop(0)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class SAGWikiEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name) / "workspace"
        self.engine = SAGWikiEngine(str(self.workspace), offline=True)
        (self.engine.raw_dir / "travel_policy_2026.md").write_text(
            "2026年现行差旅规定：上海住宿费上限为每晚650元。"
            "飞行时长超过3小时可申请经济舱全价票，须提前在OA系统审批。",
            encoding="utf-8",
        )
        (self.engine.raw_dir / "travel_policy_2024_old.md").write_text(
            "【已废止】2024年旧版规定：上海住宿费上限为每晚500元。"
            "本规定自2026年1月1日起作废。",
            encoding="utf-8",
        )
        (self.engine.raw_dir / "expense_guide.md").write_text(
            "住宿报销材料须包括住宿发票和付款记录。餐饮补贴为每人每天100元。",
            encoding="utf-8",
        )
        self.engine.compile_wiki_and_index(force_rebuild=True)

    def use_llm(self, responses: list[str]) -> FakeClient:
        client = FakeClient(responses)
        self.engine._client = client
        self.engine._offline = False
        return client

    def test_offline_pipeline_creates_all_three_layers(self) -> None:
        self.assertEqual(len(self.engine.raw_chunks), 3)
        self.assertEqual(len(self.engine.wiki_pages), 3)
        self.assertTrue(self.engine.index_path.is_file())
        self.assertEqual(len(list(self.engine.wiki_dir.glob("*.md"))), 3)
        for page in self.engine.wiki_pages.values():
            self.assertTrue(page.linked_raw_ids)
            self.assertTrue(any(raw_id in page.summary for raw_id in page.linked_raw_ids))

    def test_valid_cache_is_reused(self) -> None:
        before = json.loads(self.engine.index_path.read_text(encoding="utf-8"))
        self.engine.compile_wiki_and_index()
        after = json.loads(self.engine.index_path.read_text(encoding="utf-8"))
        self.assertEqual(before["compiled_at"], after["compiled_at"])
        self.assertEqual(before["wiki_pages"], after["wiki_pages"])

    def test_raw_change_invalidates_cache(self) -> None:
        before = json.loads(self.engine.index_path.read_text(encoding="utf-8"))
        (self.engine.raw_dir / "expense_guide.md").write_text(
            "住宿报销须提供电子发票和付款记录。餐饮补贴为每人每天120元。",
            encoding="utf-8",
        )
        self.engine.compile_wiki_and_index()
        after = json.loads(self.engine.index_path.read_text(encoding="utf-8"))
        self.assertNotEqual(before["raw_fingerprint"], after["raw_fingerprint"])
        self.assertIn("120", self.engine.wiki_pages["wiki_expense_guide"].summary)

    def test_chunk_config_change_invalidates_cache(self) -> None:
        original_size = CONFIG["raw_chunk_size"]
        original_overlap = CONFIG["raw_chunk_overlap"]
        self.addCleanup(CONFIG.__setitem__, "raw_chunk_size", original_size)
        self.addCleanup(CONFIG.__setitem__, "raw_chunk_overlap", original_overlap)
        CONFIG["raw_chunk_size"] = 20
        CONFIG["raw_chunk_overlap"] = 5
        self.engine.compile_wiki_and_index()
        data = json.loads(self.engine.index_path.read_text(encoding="utf-8"))
        self.assertEqual(data["chunking"], {"size": 20, "overlap": 5})
        self.assertGreater(len(self.engine.raw_chunks), 3)

    def test_malformed_cache_is_rebuilt(self) -> None:
        self.engine.index_path.write_text("{broken", encoding="utf-8")
        self.engine.compile_wiki_and_index()
        data = json.loads(self.engine.index_path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 2)
        self.assertTrue(data["wiki_pages"])

    def test_invalid_chunk_configuration_is_rejected(self) -> None:
        original = CONFIG["raw_chunk_overlap"]
        self.addCleanup(CONFIG.__setitem__, "raw_chunk_overlap", original)
        CONFIG["raw_chunk_overlap"] = CONFIG["raw_chunk_size"]
        with self.assertRaises(ValueError):
            self.engine.load_and_chunk_raw()

    def test_duplicate_document_stems_get_unique_chunk_ids(self) -> None:
        workspace = Path(self.temporary.name) / "collisions"
        engine = SAGWikiEngine(str(workspace), offline=True)
        (engine.raw_dir / "policy.md").write_text("Markdown policy", encoding="utf-8")
        (engine.raw_dir / "policy.txt").write_text("Text policy", encoding="utf-8")
        engine.load_and_chunk_raw()
        self.assertEqual(len(engine.raw_chunks), 2)
        self.assertEqual(len(set(engine.raw_chunks)), 2)

    def test_retrieval_prefers_current_policy(self) -> None:
        _, hits = self.engine.dual_layer_retrieve("2026年上海住宿费上限是多少？")
        self.assertTrue(hits)
        self.assertEqual(hits[0][0].chunk_id, "travel_policy_2026_c00")
        self.assertIn("650", hits[0][0].content)

    def test_cross_document_retrieval(self) -> None:
        _, hits = self.engine.dual_layer_retrieve("上海住宿费上限和报销材料是什么？")
        retrieved = {chunk.chunk_id for chunk, _ in hits}
        self.assertIn("travel_policy_2026_c00", retrieved)
        self.assertIn("expense_guide_c00", retrieved)

    def test_unanswerable_query_triggers_circuit_breaker(self) -> None:
        result = self.engine.answer("公司是否提供宠物寄养补贴？")
        self.assertEqual(result["answer"], REFUSAL)
        self.assertEqual(result["cited_ids"], [])
        self.assertEqual(result["retrieved_raw_ids"], [])

    def test_offline_answer_labels_itself_and_cites_evidence(self) -> None:
        result = self.engine.answer("上海住宿费上限是多少？")
        self.assertTrue(result["offline"])
        self.assertIn("仅返回原始证据", result["answer"])
        self.assertTrue(result["cited_ids"])
        for line in result["answer"].splitlines()[1:]:
            self.assertRegex(line, r"\[[^\]]+\]$")

    def test_llm_compiler_filters_fake_ids_and_unsafe_topic_ids(self) -> None:
        response = json.dumps(
            [
                {
                    "topic_id": "../escape",
                    "title": "Unsafe",
                    "summary": "Unsafe [travel_policy_2026_c00]",
                    "linked_raw_ids": ["travel_policy_2026_c00"],
                    "keywords": ["unsafe"],
                },
                {
                    "topic_id": "wiki_travel",
                    "title": "差旅规定",
                    "summary": "上海住宿上限为650元 [travel_policy_2026_c00]，虚构内容 [missing_c99]。",
                    "linked_raw_ids": ["travel_policy_2026_c00", "missing_c99"],
                    "keywords": ["住宿", "差旅"],
                },
            ],
            ensure_ascii=False,
        )
        self.use_llm([response])
        self.engine.compile_wiki_and_index(force_rebuild=True)
        self.assertEqual(set(self.engine.wiki_pages), {"wiki_travel"})
        page = self.engine.wiki_pages["wiki_travel"]
        self.assertEqual(page.linked_raw_ids, ["travel_policy_2026_c00"])
        self.assertNotIn("missing_c99", page.summary)
        self.assertFalse((self.engine.workspace / "escape.md").exists())

    def test_answer_rejects_hallucinated_citation(self) -> None:
        self.use_llm(["上海住宿费上限为650元 [missing_c00]。"])
        result = self.engine.answer("上海住宿费上限是多少？")
        self.assertEqual(result["answer"], REFUSAL)
        self.assertEqual(result["cited_ids"], [])

    def test_answer_rejects_uncited_claim(self) -> None:
        self.use_llm(["上海住宿费上限为650元 [travel_policy_2026_c00]。须提前审批。"])
        result = self.engine.answer("上海住宿费上限是多少？")
        self.assertEqual(result["answer"], REFUSAL)

    def test_answer_rejects_unretrieved_evidence_id(self) -> None:
        self.use_llm(["餐饮补贴为100元 [expense_guide_c00]。"])
        result = self.engine.answer("上海住宿费上限是多少？")
        self.assertEqual(result["answer"], REFUSAL)

    def test_answer_accepts_fully_grounded_response(self) -> None:
        self.use_llm(["2026年上海住宿费上限为每晚650元 [travel_policy_2026_c00]。"])
        result = self.engine.answer("上海住宿费上限是多少？")
        self.assertIn("650", result["answer"])
        self.assertEqual(result["cited_ids"], ["travel_policy_2026_c00"])

    def test_missing_wiki_file_invalidates_cache(self) -> None:
        wiki_file = next(self.engine.wiki_dir.glob("*.md"))
        wiki_file.unlink()
        self.engine.compile_wiki_and_index()
        self.assertTrue(wiki_file.is_file())

    def test_offline_cache_is_upgraded_when_llm_is_configured(self) -> None:
        response = json.dumps(
            [
                {
                    "topic_id": "wiki_travel",
                    "title": "差旅规定",
                    "summary": "上海住宿上限为650元 [travel_policy_2026_c00]。",
                    "linked_raw_ids": ["travel_policy_2026_c00"],
                    "keywords": ["住宿"],
                }
            ],
            ensure_ascii=False,
        )
        client = self.use_llm([response])
        self.engine.compile_wiki_and_index()
        data = json.loads(self.engine.index_path.read_text(encoding="utf-8"))
        self.assertEqual(data["compiler"], "llm")
        self.assertEqual(len(client.chat.completions.calls), 1)
        self.assertEqual(
            [path.name for path in self.engine.wiki_dir.glob("*.md")],
            ["wiki_travel.md"],
        )

    def test_force_rebuild_removes_previously_generated_files(self) -> None:
        data = json.loads(self.engine.index_path.read_text(encoding="utf-8"))
        stale_file = self.engine.wiki_dir / "wiki_stale.md"
        stale_file.write_text("stale", encoding="utf-8")
        data["generated_wiki_files"].append(stale_file.name)
        self.engine.index_path.write_text(json.dumps(data), encoding="utf-8")
        self.engine.compile_wiki_and_index(force_rebuild=True)
        self.assertFalse(stale_file.exists())

    def test_malformed_cached_page_entries_are_rebuilt(self) -> None:
        data = json.loads(self.engine.index_path.read_text(encoding="utf-8"))
        data["wiki_pages"] = {"wiki_bad": "not-a-page"}
        self.engine.index_path.write_text(json.dumps(data), encoding="utf-8")
        self.engine.compile_wiki_and_index()
        self.assertEqual(len(self.engine.wiki_pages), 3)

    def test_invalid_retrieval_weight_is_rejected(self) -> None:
        original = CONFIG["bm25_weight"]
        self.addCleanup(CONFIG.__setitem__, "bm25_weight", original)
        CONFIG["bm25_weight"] = 1.5
        with self.assertRaises(ValueError):
            self.engine.load_and_chunk_raw()

    def test_refusal_mixed_with_facts_is_rejected(self) -> None:
        self.use_llm(["未找到相关依据，住宿上限为999元 [travel_policy_2026_c00]。"])
        result = self.engine.answer("上海住宿费上限是多少？")
        self.assertEqual(result["answer"], REFUSAL)
        self.assertEqual(result["cited_ids"], [])

    def test_refusal_with_extra_uncited_claim_is_rejected(self) -> None:
        self.use_llm([f"{REFUSAL}住宿费上限为650元。"])
        result = self.engine.answer("上海住宿费上限是多少？")
        self.assertEqual(result["answer"], REFUSAL)
        self.assertEqual(result["cited_ids"], [])


if __name__ == "__main__":
    unittest.main()
