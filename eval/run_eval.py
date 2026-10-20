"""Experimental setup + baseline comparison.

Runs the DirectBaseline and the SemanticAssuranceAgent over the labeled dataset
and reports precision / recall / F1 / accuracy on the `should_flag` ground
truth, the confusion matrices, and the extra deviations the semantic chain
catches that the direct comparison misses.

Run:  python -m eval.run_eval
"""
from __future__ import annotations
from collections import Counter

from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             confusion_matrix)

from saa.embedding import SemanticEngine
from saa.baseline import DirectBaseline
from saa.assurance import SemanticAssuranceAgent
from saa.pipeline import canonicalize_intent, verbalize_action
from data.dataset import build_dataset


def _score(name, preds, truth):
    """Metrics via sklearn so they are standard and independently verifiable."""
    p, r, f1, _ = precision_recall_fscore_support(
        truth, preds, average="binary", pos_label=True, zero_division=0)
    acc = accuracy_score(truth, preds)
    # labels=[True, False] -> rows/cols ordered positive, negative
    (tp, fn), (fp, tn) = confusion_matrix(truth, preds, labels=[True, False])
    print(f"\n{name}")
    print(f"  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}  accuracy={acc:.3f}")
    print(f"  confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "f1": f1, "recall": r, "precision": p}


def main():
    ds = build_dataset()
    corpus = []
    for r in ds:
        corpus += [r.intent.text, canonicalize_intent(r.intent),
                   verbalize_action(r.action)]
    engine = SemanticEngine(corpus=corpus)
    pos = sum(1 for r in ds if r.should_flag)
    neg = len(ds) - pos
    print("=" * 68)
    print(f"Semantic engine : {engine.name}")
    print(f"Dataset         : {len(ds)} labeled records "
          f"({pos} positive / {neg} negative, synthetic)")
    print(f"  by case       : {dict(Counter(r.case for r in ds))}")
    print(f"Metrics         : sklearn.metrics (binary, pos_label=flag)")
    print("=" * 68)

    base = DirectBaseline(engine)
    saa = SemanticAssuranceAgent(engine)

    truth = [r.should_flag for r in ds]
    base_pred, saa_pred, saa_levels = [], [], []
    for r in ds:
        base_pred.append(base.is_breach(r.intent, r.action))
        a = saa.assess(r.intent, r.action)
        saa_levels.append(a.verdict.level)
        saa_pred.append(a.verdict.level in ("review", "breach"))  # flagged for attention

    bm = _score("BASELINE (direct input/output comparison)", base_pred, truth)
    sm = _score("SEMANTIC ASSURANCE AGENT (equivalence chains)", saa_pred, truth)

    # Extra catches: real deviations SAA flags but the baseline misses.
    print("\n" + "-" * 68)
    print("Deviations the semantic chain catches that direct comparison MISSES:")
    extra = Counter()
    for r, bp, sp in zip(ds, base_pred, saa_pred):
        if r.should_flag and sp and not bp:
            extra[r.case] += 1
    for case, n in extra.most_common():
        print(f"  + {case:12} x{n}")
    print(f"  total extra true-breach catches: {sum(extra.values())}")

    # False positives the chain removes (baseline over-flags benign paraphrases).
    removed = sum(1 for r, bp, sp in zip(ds, base_pred, saa_pred)
                  if not r.should_flag and bp and not sp)
    print(f"  false positives removed vs baseline: {removed}")

    print("\n" + "-" * 68)
    print(f"F1:  baseline={bm['f1']:.3f}  ->  SAA={sm['f1']:.3f}  "
          f"(recall {bm['recall']:.3f} -> {sm['recall']:.3f})")
    print("-" * 68)

    # Sanity cross-check: sklearn F1 equals the TP/FP/FN formula by hand.
    manual = 2 * sm["tp"] / (2 * sm["tp"] + sm["fp"] + sm["fn"])
    assert abs(manual - sm["f1"]) < 1e-9, "sklearn vs manual F1 mismatch"
    print(f"(cross-check: SAA F1 by hand = {manual:.3f}, matches sklearn)")


if __name__ == "__main__":
    main()
