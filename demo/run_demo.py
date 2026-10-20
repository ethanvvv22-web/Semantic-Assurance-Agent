"""Walk a few intent->action transformations through the Semantic Assurance
Agent and show the equivalence chain, the similarities, and the verdict next to
what the direct-comparison baseline would have said.

Run:  python -m demo.run_demo
"""
from saa.embedding import SemanticEngine
from saa.assurance import SemanticAssuranceAgent
from saa.baseline import DirectBaseline
from saa.pipeline import Intent, Action


def show(saa, base, title, it, a):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    asm = saa.assess(it, a)
    c = asm.chain
    print(f"intent     : {it.text}")
    print(f"canonical  : {c.nodes[1]}")
    print(f"action(NL) : {c.nodes[2]}")
    print(f"chain sims : links={[round(x,2) for x in c.link_sims]} "
          f"reconstruct={c.reconstruct_sim:.2f} end2end={c.end_to_end:.2f}")
    v = asm.verdict
    print(f"SAA verdict: {v.level.upper():10} score={v.score}  ::  {'; '.join(v.reasons)}")
    if v.violations:
        print(f"             violations: {[x.id for x in v.violations]}")
    print(f"BASELINE   : {'FLAG' if base.is_breach(it, a) else 'pass'} (direct comparison)")


def main():
    engine = SemanticEngine()
    print(f"[semantic engine: {engine.name}]")
    saa = SemanticAssuranceAgent(engine)
    base = DirectBaseline(engine)

    it = Intent("Reduce energy on cell 42 during the night, cut transmit power "
                "by at most 6 dB but keep coverage.", "energy",
                {"cell": "42", "window": "night", "bound_db": 6})

    show(saa, base, "A. Faithful transform (should be ACCEPTABLE)",
         it, Action("POST /ran/v1/cell/42/power", "42",
                    {"txPowerDelta_db": -6}, window="night"))

    show(saa, base, "B. Bound breach - cuts 11 dB (baseline MISSES, high similarity)",
         it, Action("POST /ran/v1/cell/42/power", "42",
                    {"txPowerDelta_db": -11}, window="night"))

    show(saa, base, "C. Scope drift - acts on cell 24 (baseline catches this)",
         it, Action("POST /ran/v1/cell/24/power", "24",
                    {"txPowerDelta_db": -6}, window="night"))

    show(saa, base, "D. Over-reach - also cancels billing (baseline MISSES)",
         it, Action("POST /ran/v1/cell/42/power", "42",
                    {"txPowerDelta_db": -6, "cancel_billing": True}, window="night"))

    show(saa, base, "E. Emergency bearer disabled (hard breach)",
         it, Action("POST /ran/v1/cell/42/power", "42",
                    {"txPowerDelta_db": -6, "disable_emergency": True}, window="night"))


if __name__ == "__main__":
    main()
