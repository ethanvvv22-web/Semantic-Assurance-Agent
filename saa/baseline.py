"""Baseline detector: a direct input/output comparison.

Per the thesis brief, the baseline compares the intent (input) directly against
the action (output) without semantic equivalence chains. Concretely it does
what a naive integration would:

  1. literal entity check - are the entities named in the intent (cell/account)
     the ones the action actually targets? and
  2. a single-hop similarity check - does the action read roughly like the intent?

This catches gross mismatches (e.g. acting on the wrong cell) but is blind to
constraint-level semantic breaches where the action still references the right
entities and reads like the intent (bound exceeded, time window dropped,
unrequested side effects, emergency bearer disabled).
"""
from __future__ import annotations

from . import config
from .pipeline import Intent, Action, verbalize_action
from .embedding import SemanticEngine


class DirectBaseline:
    def __init__(self, engine: SemanticEngine, threshold: float | None = None):
        self.engine = engine
        self.threshold = config.SIM_REVIEW if threshold is None else threshold

    def is_breach(self, it: Intent, a: Action) -> bool:
        # 1. literal entity/slot containment (input vs output)
        entity = str(it.slots.get("cell", it.slots.get("account", "")))
        if entity and entity not in f"{a.api} {a.target}":
            return True
        # 2. one-hop direct similarity
        sim = self.engine.cosine(it.text, verbalize_action(a))
        return sim < self.threshold
