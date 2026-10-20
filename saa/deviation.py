"""Acceptable-deviation scoring.

Defines where the line sits between an acceptable paraphrase-level deviation
and a real breach, by combining:
  * hard/soft constraint violations, and
  * the semantic-preservation score from the equivalence chain.

Levels: acceptable | review | breach. Thresholds live in config.py so the
thesis can sweep them (see eval/).
"""
from __future__ import annotations
from dataclasses import dataclass, field

from . import config
from .constraints import Violation, ConstraintChecker
from .chain import ChainResult


@dataclass
class Verdict:
    level: str                 # acceptable | review | breach
    score: float               # semantic-preservation score in [0,1]
    reasons: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)

    @property
    def is_breach(self) -> bool:
        return self.level == "breach"


def classify(violations: list[Violation], chain: ChainResult,
             sim_accept: float | None = None,
             sim_review: float | None = None) -> Verdict:
    sim_accept = config.SIM_ACCEPTABLE if sim_accept is None else sim_accept
    sim_review = config.SIM_REVIEW if sim_review is None else sim_review
    reasons = []

    # Semantic-preservation score: the weakest of reconstruct & end-to-end.
    score = round(min(chain.reconstruct_sim, chain.end_to_end), 3)

    if ConstraintChecker.has_hard(violations):
        reasons.append("hard constraint violated: " +
                       ", ".join(v.id for v in violations if v.severity == "hard"))
        return Verdict("breach", score, reasons, violations)

    soft = [v for v in violations if v.severity == "soft"]
    if score < sim_review:
        reasons.append(f"semantic drift: action no longer means the intent (sim={score})")
        return Verdict("breach", score, reasons, violations)

    if score < sim_accept or soft:
        if soft:
            reasons.append("soft deviation(s): " + ", ".join(v.id for v in soft))
        if score < sim_accept:
            reasons.append(f"moderate semantic distance (sim={score})")
        return Verdict("review", score, reasons, violations)

    reasons.append(f"intent preserved (sim={score}), no violations")
    return Verdict("acceptable", score, reasons, violations)
