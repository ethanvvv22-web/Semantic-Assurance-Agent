"""The Semantic Assurance Agent.

Combines the full constraint catalog, the semantic equivalence chain
(canonicalization + reconstruction), and the graded acceptable-deviation
classifier into one verdict for an (intent, action) transformation.
"""
from __future__ import annotations
from dataclasses import dataclass

from .pipeline import Intent, Action
from .embedding import SemanticEngine
from .constraints import ConstraintChecker
from .chain import EquivalenceChain, ChainResult
from .deviation import classify, Verdict


@dataclass
class Assessment:
    verdict: Verdict
    chain: ChainResult


class SemanticAssuranceAgent:
    def __init__(self, engine: SemanticEngine):
        self.engine = engine
        self.checker = ConstraintChecker()
        self.chain = EquivalenceChain(engine)

    def assess(self, it: Intent, a: Action,
               sim_accept: float | None = None,
               sim_review: float | None = None) -> Assessment:
        violations = self.checker.check(it, a)
        chain = self.chain.build(it, a)
        verdict = classify(violations, chain, sim_accept, sim_review)
        return Assessment(verdict=verdict, chain=chain)

    def is_breach(self, it: Intent, a: Action,
                  sim_accept: float | None = None,
                  sim_review: float | None = None) -> bool:
        return self.assess(it, a, sim_accept, sim_review).verdict.is_breach
