import unittest

from evaluation.significance import holm_adjust, paired_comparison


class TestPairedComparison(unittest.TestCase):
    def test_identical_systems_not_significant(self):
        a = [0.1, 0.5, 0.9, 0.3, 0.7] * 10
        res = paired_comparison(a, a)
        self.assertEqual(res.mean_diff, 0.0)
        self.assertEqual(res.p_value, 1.0)
        self.assertFalse(res.significant)
        self.assertEqual((res.ci_low, res.ci_high), (0.0, 0.0))

    def test_consistent_improvement_is_significant(self):
        a = [0.2, 0.4, 0.1, 0.5, 0.3] * 10
        b = [x + 0.2 for x in a]
        res = paired_comparison(a, b)
        self.assertAlmostEqual(res.mean_diff, 0.2)
        self.assertLess(res.p_value, 0.001)
        self.assertTrue(res.significant)
        self.assertLessEqual(res.ci_low, 0.2 + 1e-9)
        self.assertGreaterEqual(res.ci_high, 0.2 - 1e-9)

    def test_single_query_change_on_50_is_not_significant(self):
        a = [0.5] * 50
        b = [0.5] * 49 + [1.0]
        res = paired_comparison(a, b)
        self.assertAlmostEqual(res.mean_diff, 0.01)
        self.assertFalse(res.significant)
        self.assertLessEqual(res.ci_low, 0.0)

    def test_symmetric_and_deterministic(self):
        a = [0.1, 0.9, 0.4, 0.6, 0.3, 0.8]
        b = [0.3, 0.8, 0.7, 0.9, 0.2, 1.0]
        fwd, rev = paired_comparison(a, b, seed=7), paired_comparison(b, a, seed=7)
        self.assertAlmostEqual(fwd.mean_diff, -rev.mean_diff)
        self.assertEqual(fwd.p_value, rev.p_value)
        self.assertEqual(fwd, paired_comparison(a, b, seed=7))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            paired_comparison([0.1, 0.2], [0.1])
        with self.assertRaises(ValueError):
            paired_comparison([], [])
        with self.assertRaises(ValueError):
            paired_comparison([0.1], [0.2], n_resamples=0)


class TestHolmAdjust(unittest.TestCase):
    def test_known_values(self):
        # Sorted p: 0.01, 0.02, 0.04 -> 3*0.01, max(that, 2*0.02), max(that, 1*0.04)
        adjusted = holm_adjust([0.04, 0.01, 0.02])
        for got, want in zip(adjusted, [0.04, 0.03, 0.04]):
            self.assertAlmostEqual(got, want)

    def test_capped_at_one_and_monotone(self):
        adjusted = holm_adjust([0.5, 0.6, 0.001])
        self.assertEqual(adjusted[0], 1.0)
        self.assertEqual(adjusted[1], 1.0)
        self.assertAlmostEqual(adjusted[2], 0.003)

    def test_never_below_raw_p(self):
        ps = [0.03, 0.2, 0.0001, 0.049]
        for raw, adj in zip(ps, holm_adjust(ps)):
            self.assertGreaterEqual(adj, raw)

    def test_empty(self):
        self.assertEqual(holm_adjust([]), [])


if __name__ == "__main__":
    unittest.main()
