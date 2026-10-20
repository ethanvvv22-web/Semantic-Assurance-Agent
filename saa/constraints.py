"""Guardrail / constraint evaluation over an (intent, action) pair.

Loads the declarative catalog from constraints.yaml; each constraint id maps to
a predicate that returns True when VIOLATED. This is the enforcement layer and
also the part the direct-comparison baseline relies on.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

from .pipeline import Intent, Action

CATALOG = Path(__file__).with_name("constraints.yaml")


@dataclass
class Violation:
    id: str
    severity: str
    description: str


def _load():
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))["constraints"]


# predicate(intent, action) -> True if the constraint is VIOLATED
_PREDS = {
    "emergency-bearer-protected":
        lambda it, a: bool(a.params.get("disable_emergency"))
        or any("emergency" in str(s).lower() for s in a.side_effects),
    "scope-must-match":
        lambda it, a: str(a.target) != str(
            it.slots.get("cell", it.slots.get("account", a.target))),
    "power-within-bound":
        lambda it, a: ("txPowerDelta_db" in a.params and it.kind == "energy"
                       and abs(a.params["txPowerDelta_db"]) > it.slots.get("bound_db", 0)),
    "temporal-window-preserved":
        lambda it, a: (it.slots.get("window") is not None
                       and a.window != it.slots.get("window")),
    "no-unrequested-billing":
        lambda it, a: bool(a.params.get("cancel_billing")) and it.kind != "billing",
    "no-unrequested-handover":
        lambda it, a: bool(a.params.get("handover")) and it.kind != "offload",
}


class ConstraintChecker:
    def __init__(self):
        self.catalog = _load()

    def check(self, it: Intent, a: Action) -> list[Violation]:
        out = []
        for c in self.catalog:
            pred = _PREDS.get(c["id"])
            if pred is None:
                continue
            try:
                if pred(it, a):
                    out.append(Violation(c["id"], c["severity"], c["description"]))
            except Exception:
                continue
        return out

    @staticmethod
    def has_hard(violations: list[Violation]) -> bool:
        return any(v.severity == "hard" for v in violations)
