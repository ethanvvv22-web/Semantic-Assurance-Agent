"""Semantic equivalence chains.

Builds the chain  intent_text -> canonical_intent -> verbalized_action  and
measures semantic similarity along each link. The backward link
(canonical_intent vs verbalized_action) is the reconstruct-the-intent test:
if what the action actually does no longer means what was asked, the link
similarity drops - catching deviations a direct input/output match misses.
"""
from __future__ import annotations
from dataclasses import dataclass

from .pipeline import Intent, Action, canonicalize_intent, verbalize_action
from .embedding import SemanticEngine


@dataclass
class ChainResult:
    nodes: list[str]
    link_sims: list[float]      # adjacent-node similarities
    end_to_end: float           # intent_text vs verbalized_action
    reconstruct_sim: float      # canonical_intent vs verbalized_action
    min_link: float


class EquivalenceChain:
    def __init__(self, engine: SemanticEngine):
        self.engine = engine

    def build(self, it: Intent, a: Action) -> ChainResult:
        n0 = it.text
        n1 = canonicalize_intent(it)
        n2 = verbalize_action(a)
        nodes = [n0, n1, n2]
        links = [self.engine.cosine(nodes[i], nodes[i + 1])
                 for i in range(len(nodes) - 1)]
        e2e = self.engine.cosine(n0, n2)
        recon = self.engine.cosine(n1, n2)
        return ChainResult(nodes=nodes, link_sims=links, end_to_end=e2e,
                           reconstruct_sim=recon, min_link=min(links + [e2e]))
