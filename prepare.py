"""Evaluate a SAG-Wiki engine against a fixed, citation-aware golden set."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "fixtures" / "golden.jsonl"
RAW_FIXTURES_PATH = PROJECT_ROOT / "fixtures" / "raw"
REFUSAL = "当前知识库中未找到相关依据。"
CITATION_RE = re.compile(r"\[([\w-]+)\]", re.UNICODE)


def load_engine(engine_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("candidate_sag_wiki_engine", engine_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load engine from {engine_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SAGWikiEngine


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(case, dict):
            raise ValueError(f"{path}:{line_number}: case must be an object")
        case_id = case.get("id")
        question = case.get("question")
        expected_ids = case.get("expected_raw_ids")
        required_facts = case.get("required_facts")
        expect_refusal = case.get("expect_refusal")
        if not all(isinstance(value, str) and value.strip() for value in (case_id, question)):
            raise ValueError(f"{path}:{line_number}: id and question must be non-empty strings")
        if case_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate case id {case_id}")
        if not isinstance(expected_ids, list) or not all(
            isinstance(value, str) for value in expected_ids
        ):
            raise ValueError(f"{path}:{line_number}: expected_raw_ids must be a string list")
        if not isinstance(required_facts, list) or not all(
            isinstance(value, str) and value for value in required_facts
        ):
            raise ValueError(f"{path}:{line_number}: required_facts must be a string list")
        if not isinstance(expect_refusal, bool):
            raise ValueError(f"{path}:{line_number}: expect_refusal must be boolean")
        if expect_refusal and expected_ids:
            raise ValueError(f"{path}:{line_number}: refusal cases cannot expect evidence")
        seen_ids.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError("Golden dataset is empty")
    return cases


def normalise(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def retrieval_metrics(
    retrieved_ids: List[str],
    expected_ids: List[str],
) -> tuple[float, float]:
    if not expected_ids:
        no_false_positive = float(not retrieved_ids)
        return no_false_positive, no_false_positive
    ranks = [
        retrieved_ids.index(raw_id) + 1
        for raw_id in expected_ids
        if raw_id in retrieved_ids
    ]
    hit = 1.0 if ranks else 0.0
    reciprocal_rank = 1.0 / min(ranks) if ranks else 0.0
    return hit, reciprocal_rank


def evaluate_case(
    engine: Any,
    case: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    started = time.perf_counter()
    _, raw_hits = engine.dual_layer_retrieve(case["question"])
    retrieved_ids = [chunk.chunk_id for chunk, _ in raw_hits]
    hit, reciprocal_rank = retrieval_metrics(retrieved_ids, case["expected_raw_ids"])

    citation_valid = None
    answer_correct = None
    answer_text = ""
    cited_ids: List[str] = []
    if mode == "full":
        result = engine.answer(case["question"])
        answer_text = result["answer"]
        reported_ids = result.get("cited_ids", [])
        if not isinstance(reported_ids, list) or not all(
            isinstance(raw_id, str) for raw_id in reported_ids
        ):
            reported_ids = []
        cited_ids = list(dict.fromkeys(CITATION_RE.findall(answer_text)))
        retrieved_set = set(retrieved_ids)
        refusal_only = (
            answer_text.strip().rstrip("。！？.!? \n") == REFUSAL.rstrip("。！？.!? \n")
        )
        if case["expect_refusal"]:
            citation_valid = float(not cited_ids and not reported_ids)
            answer_correct = float(
                refusal_only
                and not cited_ids
                and not reported_ids
                and not retrieved_ids
            )
        else:
            citation_valid = float(
                bool(cited_ids)
                and cited_ids == reported_ids
                and set(cited_ids) <= retrieved_set
            )
            facts_present = all(
                normalise(fact) in normalise(answer_text)
                for fact in case["required_facts"]
            )
            answer_correct = float(citation_valid and facts_present)

    if mode == "retrieval":
        score = 0.7 * hit + 0.3 * reciprocal_rank
    else:
        score = (
            0.35 * hit
            + 0.15 * reciprocal_rank
            + 0.20 * citation_valid
            + 0.30 * answer_correct
        )

    return {
        "case_id": case["id"],
        "retrieval_hit": hit,
        "reciprocal_rank": reciprocal_rank,
        "citation_valid": citation_valid,
        "answer_correct": answer_correct,
        "score": score,
        "latency_sec": round(time.perf_counter() - started, 3),
        "retrieved_raw_ids": retrieved_ids,
        "cited_ids": cited_ids,
        "answer": answer_text,
        "error": "",
    }


def error_row(case_id: str, error: Exception) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "retrieval_hit": 0.0,
        "reciprocal_rank": 0.0,
        "citation_valid": 0.0,
        "answer_correct": 0.0,
        "score": 0.0,
        "latency_sec": 0.0,
        "retrieved_raw_ids": [],
        "cited_ids": [],
        "answer": "",
        "error": f"{type(error).__name__}: {error}",
    }


def write_results(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case_id",
        "retrieval_hit",
        "reciprocal_rank",
        "citation_valid",
        "answer_correct",
        "score",
        "latency_sec",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("retrieval", "full"),
        default="retrieval",
        help="retrieval is offline; full also evaluates grounded answers",
    )
    parser.add_argument("--engine", type=Path, default=PROJECT_ROOT / "sag_wiki_engine.py")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results.tsv")
    parser.add_argument("--min-score", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "full" and not os.getenv("OPENAI_API_KEY"):
        print("full mode requires OPENAI_API_KEY", file=sys.stderr)
        return 2
    if args.min_score is not None and not 0.0 <= args.min_score <= 1.0:
        print("--min-score must be between 0 and 1", file=sys.stderr)
        return 2

    try:
        engine_class = load_engine(args.engine.resolve())
        cases = load_dataset(DATASET_PATH)
        if not RAW_FIXTURES_PATH.is_dir():
            raise ValueError(f"Missing raw fixtures: {RAW_FIXTURES_PATH}")
    except (OSError, ValueError, ImportError) as error:
        print(f"Evaluation setup failed: {error}", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sag-wiki-eval-") as temporary:
        workspace = Path(temporary) / "workspace"
        raw_dir = workspace / "raw"
        raw_dir.mkdir(parents=True)
        for source in sorted(RAW_FIXTURES_PATH.iterdir()):
            if source.is_file() and source.suffix in {".md", ".txt"}:
                shutil.copy2(source, raw_dir / source.name)

        try:
            engine = engine_class(str(workspace), offline=args.mode == "retrieval")
            engine.compile_wiki_and_index(force_rebuild=True)
        except Exception as error:
            print(f"Engine initialisation failed: {error}", file=sys.stderr)
            return 2

        for case in cases:
            try:
                rows.append(evaluate_case(engine, case, args.mode))
            except Exception as error:
                rows.append(error_row(case["id"], error))

    write_results(args.output, rows)
    count = len(rows)
    summary = {
        "mode": args.mode,
        "cases": count,
        "score": round(sum(row["score"] for row in rows) / count, 4),
        "retrieval_hit": round(
            sum(row["retrieval_hit"] for row in rows) / count, 4
        ),
        "mrr": round(sum(row["reciprocal_rank"] for row in rows) / count, 4),
        "errors": sum(bool(row["error"]) for row in rows),
        "output": str(args.output),
    }
    if args.mode == "full":
        summary["citation_valid"] = round(
            sum(row["citation_valid"] for row in rows) / count, 4
        )
        summary["answer_correct"] = round(
            sum(row["answer_correct"] for row in rows) / count, 4
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in rows:
        if row["error"]:
            print(f"{row['case_id']}: {row['error']}", file=sys.stderr)

    if summary["errors"]:
        return 1
    if args.min_score is not None and summary["score"] < args.min_score:
        print(
            f"Score {summary['score']:.4f} is below required {args.min_score:.4f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
