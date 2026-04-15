"""
Composite Query Planner
-----------------------
Chains multiple planners so that each one observes the cfg produced by the
previous one. Sub-planners are applied in the order given; `expand_queries`
returns the first sub-planner's non-trivial expansion (len > 1), otherwise
falls back to the original query.

Example:
    CompositeQueryPlanner(
        cfg,
        [MultiHopQueryPlanner(cfg), HeuristicQueryPlanner(cfg)],
    )

    - MultiHopQueryPlanner runs first: may widen num_candidates and yield
      sub-questions.
    - HeuristicQueryPlanner runs second on the already-widened cfg and
      overrides ranker_weights based on query classification.
"""
from __future__ import annotations

from copy import deepcopy
from typing import List, Sequence

from src.config import RAGConfig
from src.planning.planner import QueryPlanner


class CompositeQueryPlanner(QueryPlanner):
    def __init__(self, base_cfg: RAGConfig, sub_planners: Sequence[QueryPlanner]):
        super().__init__(base_cfg)
        if not sub_planners:
            raise ValueError("CompositeQueryPlanner requires at least one sub-planner.")
        self.sub_planners: List[QueryPlanner] = list(sub_planners)

    @property
    def name(self) -> str:
        inner = " -> ".join(p.name for p in self.sub_planners)
        return f"Composite[{inner}]"

    def plan(self, query: str) -> RAGConfig:
        cfg = deepcopy(self.base_cfg)
        for sub in self.sub_planners:
            # Feed the in-progress cfg into the next sub-planner by temporarily
            # swapping its base_cfg; restore it afterwards so the sub-planner
            # remains reusable outside this composite.
            saved = sub.base_cfg
            sub.base_cfg = cfg
            try:
                cfg = sub.plan(query)
            finally:
                sub.base_cfg = saved
        return cfg

    def expand_queries(self, query: str) -> List[str]:
        for sub in self.sub_planners:
            expanded = sub.expand_queries(query)
            if expanded and len(expanded) > 1:
                return expanded
        return [query]
