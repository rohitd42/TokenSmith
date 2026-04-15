#!/usr/bin/env python3
"""Print top-3 retrieved chunks for a fixed list of two-phase locking queries."""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RAGConfig
from src.instrumentation.logging import get_logger
from src.main import get_answer
from src.planning.composite import CompositeQueryPlanner
from src.planning.heuristics import HeuristicQueryPlanner
from src.planning.multihop import MultiHopQueryPlanner
from src.ranking.ranker import EnsembleRanker
from src.retriever import BM25Retriever, FAISSRetriever, load_artifacts


INDEX_PREFIX = "textbook_index"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
ARTIFACTS_DIR = REPO_ROOT / "index" / "sections"

QUERIES = [
    "What is the two-phase locking protocol?",
    "What are the two phases in two-phase locking?",
    "What is the difference between strict and rigorous two-phase locking?",
    "How does two-phase locking guarantee serializability?",
    "What is lock point in two-phase locking and why does it matter?",
]

TOP_N = 3


def build_artifacts(cfg: RAGConfig) -> Dict[str, Any]:
    faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(
        ARTIFACTS_DIR, INDEX_PREFIX
    )
    retrievers = [
        FAISSRetriever(faiss_idx, cfg.embed_model),
        BM25Retriever(bm25_idx),
    ]
    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )
    planner = CompositeQueryPlanner(
        cfg, [MultiHopQueryPlanner(cfg), HeuristicQueryPlanner(cfg)]
    )
    return {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "meta": meta,
        "planner": planner,
    }


def main() -> None:
    cfg = RAGConfig.from_yaml(CONFIG_PATH)
    logger = get_logger()
    args = SimpleNamespace(
        system_prompt_mode="baseline",
        double_prompt=False,
        index_prefix=INDEX_PREFIX,
    )

    print(f"Loading artifacts from {ARTIFACTS_DIR} ...")
    artifacts = build_artifacts(cfg)

    for i, query in enumerate(QUERIES, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(QUERIES)}] QUERY: {query}")
        print("=" * 80)

        result = get_answer(
            question=query,
            cfg=cfg,
            args=args,
            logger=logger,
            console=None,
            artifacts=artifacts,
            is_test_mode=True,
        )
        if not isinstance(result, tuple):
            print(f"[ANSWER_NOT_FOUND path] result={result!r}")
            continue

        _ans, chunks_info, _ = result
        chunks_info = chunks_info or []
        if not chunks_info:
            print("(no chunks retrieved)")
            continue

        for rank, ch in enumerate(chunks_info[:TOP_N], 1):
            content = str(ch.get("content", ""))
            src = ch.get("source") or ch.get("sources") or ""
            pages = ch.get("page_numbers") or ch.get("pages") or ""
            print(f"\n--- chunk {rank} (source={src} pages={pages}) ---")
            print(content)


if __name__ == "__main__":
    main()
