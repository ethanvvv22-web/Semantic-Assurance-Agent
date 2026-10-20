"""Deterministic unit + integration tests (stdlib unittest).

Uses the TF-IDF backend so tests are fast, offline and reproducible.
Run:  python -m unittest discover -s tests
"""
import unittest

from saa.pipeline import Intent, Action, verbalize_action, canonicalize_intent
from saa.constraints import ConstraintChecker
from saa.chain import ChainResult, EquivalenceChain
from saa.deviation import classify
from saa.embedding import SemanticEngine
from saa.assurance import SemanticAssuranceAgent
from saa.baseline import DirectBaseline

IT = Intent("Reduce energy on cell 42 at night, at most 6 dB.", "energy",
            {"cell": "42", "window": "night", "bound_db": 6})
FAITHFUL = Action("POST /ran/v1/cell/42/power", "42",
                  {"txPowerDelta_db": -6}, window="night")


def engine():
    corpus = [IT.text, canonicalize_intent(IT), verbalize_action(FAITHFUL),
              "billing cancel account cell power night emergency"]
    return SemanticEngine(backend="tfidf", corpus=corpus)


class TestConstraints(unittest.TestCase):
    def setUp(self):
        self.c = ConstraintChecker()

    def test_faithful_clean(self):
        self.assertEqual(self.c.check(IT, FAITHFUL), [])

    def test_bound_breach(self):
        v = self.c.check(IT, Action("POST /ran/v1/cell/42/power", "42",
                                    {"txPowerDelta_db": -11}, window="night"))
        self.assertIn("power-within-bound", [x.id for x in v])
        self.assertTrue(ConstraintChecker.has_hard(v))

    def test_scope_drift(self):
        v = self.c.check(IT, Action("POST /ran/v1/cell/24/power", "24",
                                    {"txPowerDelta_db": -6}, window="night"))
        self.assertIn("scope-must-match", [x.id for x in v])

    def test_overreach_billing(self):
        v = self.c.check(IT, Action("POST /ran/v1/cell/42/power", "42",
                                    {"txPowerDelta_db": -6, "cancel_billing": True},
                                    window="night"))
        self.assertIn("no-unrequested-billing", [x.id for x in v])

    def test_emergency(self):
        v = self.c.check(IT, Action("POST /ran/v1/cell/42/power", "42",
                                    {"txPowerDelta_db": -6, "disable_emergency": True},
                                    window="night"))
        self.assertIn("emergency-bearer-protected", [x.id for x in v])

    def test_temporal_soft(self):
        v = self.c.check(IT, Action("POST /ran/v1/cell/42/power", "42",
                                    {"txPowerDelta_db": -6}, window=None))
        ids = [x.id for x in v]
        self.assertIn("temporal-window-preserved", ids)
        self.assertFalse(ConstraintChecker.has_hard(v))


class TestClassify(unittest.TestCase):
    def _chain(self, recon, e2e):
        return ChainResult(["a", "b", "c"], [0.9, recon], e2e, recon, min(0.9, recon, e2e))

    def test_hard_is_breach_regardless_of_sim(self):
        from saa.constraints import Violation
        v = classify([Violation("x", "hard", "d")], self._chain(0.99, 0.99))
        self.assertEqual(v.level, "breach")

    def test_high_sim_no_violation_acceptable(self):
        v = classify([], self._chain(0.9, 0.9))
        self.assertEqual(v.level, "acceptable")

    def test_low_sim_breach(self):
        v = classify([], self._chain(0.2, 0.2))
        self.assertEqual(v.level, "breach")

    def test_soft_violation_downgrades_to_review(self):
        from saa.constraints import Violation
        v = classify([Violation("temporal-window-preserved", "soft", "d")],
                     self._chain(0.9, 0.9))
        self.assertEqual(v.level, "review")


class TestEmbedding(unittest.TestCase):
    def test_identity_similarity(self):
        e = engine()
        self.assertAlmostEqual(e.cosine("cell 42 power", "cell 42 power"), 1.0, places=5)

    def test_backend_is_tfidf(self):
        self.assertEqual(engine().name, "tfidf")


class TestAgents(unittest.TestCase):
    def test_saa_flags_hard_breach(self):
        saa = SemanticAssuranceAgent(engine())
        self.assertTrue(saa.is_breach(IT, Action("POST /ran/v1/cell/42/power", "42",
                                                  {"txPowerDelta_db": -11}, window="night")))

    def test_saa_accepts_faithful(self):
        # thresholds set low so the test checks wiring, not embedding quality
        saa = SemanticAssuranceAgent(engine())
        self.assertFalse(saa.is_breach(IT, FAITHFUL, sim_accept=0.1, sim_review=0.05))

    def test_baseline_catches_scope_misses_bound(self):
        base = DirectBaseline(engine(), threshold=0.05)  # isolate the entity check
        self.assertTrue(base.is_breach(IT, Action("POST /ran/v1/cell/24/power", "24",
                                                  {"txPowerDelta_db": -6}, window="night")))
        self.assertFalse(base.is_breach(IT, Action("POST /ran/v1/cell/42/power", "42",
                                                   {"txPowerDelta_db": -11}, window="night")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
