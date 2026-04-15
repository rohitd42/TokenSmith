#!/usr/bin/env python3
"""
eval/run_eval.py

Evaluate the TokenSmith RAG pipeline on eval/questions.jsonl, comparing the
CompositeQueryPlanner (MultiHop -> Heuristic, the "optimizer") against a
no-op baseline planner that leaves cfg unchanged and never expands the query.

The baseline and optimizer share the same artifacts (chunks, FAISS index,
BM25 index, embedding model) — only the planner wiring differs. The eval
calls `src.main.get_answer` directly with `is_test_mode=True` so no streaming
or markdown rendering happens.

Usage:
    python -m eval.run_eval                  # run both modes (default)
    python -m eval.run_eval --baseline       # run baseline only
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path when invoked as a script
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RAGConfig
from src.instrumentation.logging import get_logger
from src.main import ANSWER_NOT_FOUND, get_answer
from src.planning.composite import CompositeQueryPlanner
from src.planning.heuristics import HeuristicQueryPlanner
from src.planning.multihop import MultiHopQueryPlanner
from src.planning.planner import QueryPlanner
from src.ranking.ranker import EnsembleRanker
from src.retriever import (
    BM25Retriever,
    FAISSRetriever,
    IndexKeywordRetriever,
    load_artifacts,
)


INDEX_PREFIX = "textbook_index"
QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"
RESULTS_PATH = REPO_ROOT / "eval" / "results.csv"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class NoOpPlanner(QueryPlanner):
    """Planner that returns the base cfg unchanged and never expands queries."""

    @property
    def name(self) -> str:
        return "NoOpPlanner"

    def plan(self, query: str):
        # deepcopy so callers can't mutate our stored base_cfg by accident
        return deepcopy(self.base_cfg)

    # expand_queries inherits the default `[query]` from QueryPlanner


def build_args() -> SimpleNamespace:
    # get_answer reads args.system_prompt_mode and (via getattr) args.double_prompt
    return SimpleNamespace(
        system_prompt_mode="baseline",
        double_prompt=False,
        index_prefix=INDEX_PREFIX,
    )


def build_artifacts(cfg: RAGConfig, planner: QueryPlanner) -> Dict[str, Any]:
    """
    Mirror run_chat_session's artifact setup but swap in an arbitrary planner.
    """
    artifacts_dir = cfg.get_artifacts_directory()
    faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(
        artifacts_dir, INDEX_PREFIX
    )
    retrievers: List[Any] = [
        FAISSRetriever(faiss_idx, cfg.embed_model),
        BM25Retriever(bm25_idx),
    ]
    if cfg.ranker_weights.get("index_keywords", 0) > 0:
        retrievers.append(
            IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path)
        )
    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )
    return {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "meta": meta,
        "planner": planner,
    }


def load_questions() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def retrieval_hit(chunks_info: Optional[List[Dict[str, Any]]], expected: List[str]) -> int:
    if not chunks_info or not expected:
        return 0
    contents = [str(c.get("content", "")).lower() for c in chunks_info]
    for needle in expected:
        n = str(needle).lower()
        if any(n in content for content in contents):
            return 1
    return 0


def answer_hit(answer_text: str, gold_fragment: str) -> int:
    if not answer_text or not gold_fragment:
        return 0
    return 1 if gold_fragment.lower() in answer_text.lower() else 0


def run_one(
    question: str,
    cfg: RAGConfig,
    artifacts: Dict[str, Any],
    args: SimpleNamespace,
    logger,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Call get_answer in test mode. Returns (answer_text, chunks_info).
    Handles the ANSWER_NOT_FOUND early-return path where get_answer yields
    a bare string instead of the tuple.
    """
    result = get_answer(
        question=question,
        cfg=cfg,
        args=args,
        logger=logger,
        console=None,
        artifacts=artifacts,
        is_test_mode=True,
    )
    if isinstance(result, tuple):
        ans, chunks_info, _ = result
        return ans or "", chunks_info or []
    # ANSWER_NOT_FOUND path — no chunks retrieved
    return str(result), []


def main() -> None:
    parser = argparse.ArgumentParser(description="TokenSmith planner evaluation")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run only the no-op baseline (skip CompositeQueryPlanner)",
    )
    parser.add_argument(
        "--optimizer",
        action="store_true",
        help="Run only the CompositeQueryPlanner (skip baseline)",
    )
    cli = parser.parse_args()

    run_baseline = cli.baseline or not cli.optimizer
    run_optimizer = cli.optimizer or not cli.baseline

    if not CONFIG_PATH.exists():
        print(f"ERROR: missing config at {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    cfg = RAGConfig.from_yaml(CONFIG_PATH)
    logger = get_logger()
    args = build_args()

    print(f"Loading artifacts from {cfg.get_artifacts_directory()} ...")

    baseline_artifacts: Optional[Dict[str, Any]] = None
    optimizer_artifacts: Optional[Dict[str, Any]] = None
    if run_baseline:
        baseline_artifacts = build_artifacts(cfg, NoOpPlanner(cfg))
    if run_optimizer:
        composite = CompositeQueryPlanner(
            cfg,
            [MultiHopQueryPlanner(cfg), HeuristicQueryPlanner(cfg)],
        )
        optimizer_artifacts = build_artifacts(cfg, composite)

    questions = load_questions()
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")

    # Use a dedicated HeuristicQueryPlanner to label the CSV regardless of
    # which modes are run. This keeps the "planner_classification" column
    # stable and lets us diagnose misclassifications even on baseline-only
    # runs.
    label_planner = HeuristicQueryPlanner(cfg)

    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        query = q["query"]
        category = q.get("category", "")
        expected = q.get("expected_chunks", [])
        gold = q.get("gold_answer_fragment", "")

        classification = label_planner.classify(query)

        b_retr: Any = ""
        o_retr: Any = ""
        b_ans: Any = ""
        o_ans: Any = ""

        print(f"\n[{i}/{len(questions)}] ({category}) {query}")

        if run_baseline and baseline_artifacts is not None:
            print("  -- baseline --")
            ans_b, chunks_b = run_one(query, cfg, baseline_artifacts, args, logger)
            b_retr = retrieval_hit(chunks_b, expected)
            b_ans = answer_hit(ans_b, gold)
            print(f"    retrieval_hit={b_retr} answer_hit={b_ans}")

        if run_optimizer and optimizer_artifacts is not None:
            print("  -- optimizer --")
            ans_o, chunks_o = run_one(query, cfg, optimizer_artifacts, args, logger)
            o_retr = retrieval_hit(chunks_o, expected)
            o_ans = answer_hit(ans_o, gold)
            print(f"    retrieval_hit={o_retr} answer_hit={o_ans}")

        rows.append({
            "query": query,
            "category": category,
            "planner_classification": classification,
            "baseline_retrieval_hit": b_retr,
            "optimizer_retrieval_hit": o_retr,
            "baseline_answer_hit": b_ans,
            "optimizer_answer_hit": o_ans,
        })

    write_results(rows)
    print_summary(rows, run_baseline, run_optimizer)


def write_results(rows: List[Dict[str, Any]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query",
        "category",
        "planner_classification",
        "baseline_retrieval_hit",
        "optimizer_retrieval_hit",
        "baseline_answer_hit",
        "optimizer_answer_hit",
    ]
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote results to {RESULTS_PATH}")


def _rate(values: List[int]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def print_summary(rows: List[Dict[str, Any]], run_baseline: bool, run_optimizer: bool) -> None:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[r.get("category") or "unknown"].append(r)

    cols = [f"{'category':<12}", f"{'n':>4}"]
    if run_baseline:
        cols += [f"{'B retr':>8}", f"{'B ans':>8}"]
    if run_optimizer:
        cols += [f"{'O retr':>8}", f"{'O ans':>8}"]
    header = " ".join(cols)

    print()
    print("=" * len(header))
    print("Hit rates by category")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    def fmt_row(label: str, items: List[Dict[str, Any]]) -> str:
        parts = [f"{label:<12}", f"{len(items):>4}"]
        if run_baseline:
            br = _rate([int(x.get("baseline_retrieval_hit") or 0) for x in items])
            ba = _rate([int(x.get("baseline_answer_hit") or 0) for x in items])
            parts += [f"{br:>8.2%}", f"{ba:>8.2%}"]
        if run_optimizer:
            orr = _rate([int(x.get("optimizer_retrieval_hit") or 0) for x in items])
            oa = _rate([int(x.get("optimizer_answer_hit") or 0) for x in items])
            parts += [f"{orr:>8.2%}", f"{oa:>8.2%}"]
        return " ".join(parts)

    for cat in sorted(buckets):
        print(fmt_row(cat, buckets[cat]))
    print("-" * len(header))
    print(fmt_row("ALL", rows))


if __name__ == "__main__":
    main()
