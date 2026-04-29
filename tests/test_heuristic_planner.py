"""Unit tests for HeuristicQueryPlanner classification.

These tests are artifact-free and safe to run in CI.
"""
from __future__ import annotations

import pytest

from src.config import RAGConfig
from src.planning.heuristics import HeuristicQueryPlanner


@pytest.fixture()
def base_cfg() -> RAGConfig:
    return RAGConfig()


@pytest.mark.parametrize(
    "query",
    [
        "how does BATMAN-adv route packets",
        "how do neural networks learn",
        "how is query optimization performed",
        "how are indexes maintained in b+ trees",
        "why does the transaction fail under contention",
        "why is the optimizer slow",
        "what causes deadlocks in two-phase locking",
        "explain the two-phase commit protocol",
    ],
)
def test_explanatory_triggers(base_cfg, query):
    p = HeuristicQueryPlanner(base_cfg)
    assert p.classify(query) == "explanatory"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("how to create an index", "procedural"),
        ("steps to recover from a crash", "procedural"),
        ("what is a transaction", "definition"),
        ("compare clustered and non-clustered indexes", "comparison"),
        ("difference between 2PL and MVCC", "keyword"),  # acronym wins
        ("what is ACID", "keyword"),
    ],
)
def test_non_explanatory_unchanged(base_cfg, query, expected):
    p = HeuristicQueryPlanner(base_cfg)
    assert p.classify(query) == expected


def test_other_is_last_fallback(base_cfg):
    p = HeuristicQueryPlanner(base_cfg)
    assert p.classify("database systems concepts") == "other"
