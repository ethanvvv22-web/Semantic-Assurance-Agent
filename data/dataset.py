"""Synthetic labeled dataset of intent -> action transformations.

Each record carries ground truth `should_flag` (True = the transformation is a
real deviation/breach that must not pass as-is). Cases are crafted so the
contrast between a direct input/output comparison and the semantic equivalence
chain is measurable:

  * baseline false-negatives: the action still reads like the intent (high
    direct similarity) yet breaks a constraint (bound / scope / over-reach).
  * baseline false-positives: a benign action phrased differently from a noisy
    intent (low direct similarity) that is actually fine.
"""
from __future__ import annotations
from dataclasses import dataclass, replace

from saa.pipeline import Intent, Action


@dataclass
class Record:
    intent: Intent
    action: Action
    should_flag: bool
    case: str          # ok | ok_paraphrase | bound | scope | temporal | overreach | emergency | handover


def _energy_intent(cell: str, wordy: bool = False) -> Intent:
    base = f"Reduce energy on cell {cell} during the night, cut transmit power by at most 6 dB but keep coverage."
    if wordy:
        base = (f"As part of our quarterly sustainability program and to lower OPEX, "
                f"please arrange that cell {cell} consumes less energy overnight; a modest "
                f"transmit-power reduction of no more than 6 dB is fine as long as users keep coverage.")
    return Intent(text=base, kind="energy",
                  slots={"cell": cell, "window": "night", "bound_db": 6})


def _energy_faithful(cell: str) -> Action:
    return Action(api=f"POST /ran/v1/cell/{cell}/power", target=cell,
                  params={"txPowerDelta_db": -6}, window="night")


def _billing_intent(acc: str) -> Intent:
    return Intent(text=f"Terminate the subscription for account {acc}.",
                  kind="billing", slots={"account": acc})


def _billing_faithful(acc: str) -> Action:
    return Action(api=f"POST /billing/v1/account/{acc}/cancel", target=acc,
                  params={"cancel_billing": True})


def build_dataset() -> list[Record]:
    recs: list[Record] = []
    cells = ["42", "7", "15", "88", "23"]
    for c in cells:
        it = _energy_intent(c)
        fa = _energy_faithful(c)
        # TN: faithful
        recs.append(Record(it, fa, False, "ok"))
        # FP-bait for baseline: wordy intent, same faithful action
        recs.append(Record(_energy_intent(c, wordy=True), fa, False, "ok_paraphrase"))
        # FN-bait: bound breach (same api/target/window -> high direct sim)
        recs.append(Record(it, replace(fa, params={"txPowerDelta_db": -11}), True, "bound"))
        # FN-bait: scope drift (acts on a different cell)
        other = "24" if c == "42" else "99"
        recs.append(Record(it, replace(fa, api=f"POST /ran/v1/cell/{other}/power",
                                        target=other), True, "scope"))
        # temporal drop (acts all day)
        recs.append(Record(it, replace(fa, window=None), True, "temporal"))
        # over-reach: energy intent that also cancels billing
        recs.append(Record(it, replace(fa, params={"txPowerDelta_db": -6,
                                                    "cancel_billing": True}), True, "overreach"))
        # emergency breach
        recs.append(Record(it, replace(fa, params={"txPowerDelta_db": -6,
                                                    "disable_emergency": True}), True, "emergency"))
    for acc in ["A1001", "A2002", "A3003"]:
        it = _billing_intent(acc)
        recs.append(Record(it, _billing_faithful(acc), False, "ok"))
        # scope drift on billing: cancels a different account
        bad = "A9999"
        recs.append(Record(it, Action(api=f"POST /billing/v1/account/{bad}/cancel",
                                       target=bad, params={"cancel_billing": True}),
                           True, "scope"))
    return recs


if __name__ == "__main__":
    ds = build_dataset()
    from collections import Counter
    print(f"{len(ds)} records:", dict(Counter(r.case for r in ds)))
    print("should_flag:", dict(Counter(r.should_flag for r in ds)))
