"""The agentic transformation under test, plus the verbalizers that turn
intents and actions into canonical natural language for the equivalence chain.

Flow (telecom / autonomous-network flavored):
  operator intent (NL)  ->  structured decision  ->  executable action (API call)

A faithful transform preserves the intent's slots and respects guardrails.
Faulty transforms (scope drift, dropped temporal qualifier, over-reach,
hard-constraint breach) are what the Semantic Assurance Agent must catch.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Intent:
    text: str
    kind: str                 # energy | priority | offload | billing
    slots: dict               # e.g. {"cell": "42", "window": "night", "bound_db": 6}


@dataclass
class Action:
    api: str                  # e.g. POST /ran/v1/cell/42/power
    target: str               # cell id / account id
    params: dict = field(default_factory=dict)
    window: str | None = None
    side_effects: list = field(default_factory=list)


# --- canonical verbalizers (the chain's natural-language nodes) -------------
def canonicalize_intent(it: Intent) -> str:
    s = it.slots
    if it.kind == "energy":
        w = s.get("window", "any time")
        return (f"save energy on cell {s.get('cell')} during {w} by reducing "
                f"transmit power up to {s.get('bound_db', '?')} dB without losing coverage")
    if it.kind == "priority":
        return f"prioritize {s.get('traffic','emergency')} traffic on cell {s.get('cell')}"
    if it.kind == "offload":
        return f"offload subscribers from congested cell {s.get('cell')} to neighbors"
    if it.kind == "billing":
        return f"terminate the subscription for account {s.get('account')}"
    return it.text


def verbalize_action(a: Action) -> str:
    """Backward chain: describe, in NL, what the action actually does."""
    parts = [f"call {a.api} on target {a.target}"]
    if "txPowerDelta_db" in a.params:
        parts.append(f"reduce transmit power by {abs(a.params['txPowerDelta_db'])} dB")
    if a.params.get("sleep"):
        parts.append("put the cell to sleep")
    if a.params.get("priority"):
        parts.append(f"set scheduling priority to {a.params['priority']}")
    if a.params.get("handover"):
        parts.append("hand subscribers over to neighboring cells")
    if a.params.get("cancel_billing"):
        parts.append("cancel billing and terminate the subscription")
    if a.window:
        parts.append(f"during {a.window}")
    else:
        parts.append("at any time")
    for se in a.side_effects:
        parts.append(f"side effect: {se}")
    return ", ".join(parts)
