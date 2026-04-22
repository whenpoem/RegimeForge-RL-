from __future__ import annotations

import unittest

import numpy as np

from backend.regime_lens.stats_ext import (
    bayesian_effect_summary,
    bootstrap_ci,
    friedman_with_posthoc,
    wilcoxon_signed_rank,
)


class StatsExtTests(unittest.TestCase):
    def test_robust_statistics_smoke(self) -> None:
        better = np.array([0.92, 0.88, 0.91, 0.9, 0.89, 0.93], dtype=np.float64)
        worse = np.array([0.55, 0.52, 0.57, 0.54, 0.56, 0.53], dtype=np.float64)

        ci_low, ci_high = bootstrap_ci(better, n_resamples=1_000, seed=123)
        self.assertLess(ci_low, ci_high)
        self.assertLessEqual(ci_low, float(np.mean(better)))
        self.assertGreaterEqual(ci_high, float(np.mean(better)))

        wilcoxon = wilcoxon_signed_rank(better, worse)
        self.assertEqual(wilcoxon["n"], len(better))
        self.assertLess(wilcoxon["p_value"], 0.1)

        friedman = friedman_with_posthoc(
            {
                "rcmoe": better,
                "dqn": np.array([0.71, 0.68, 0.7, 0.69, 0.72, 0.67], dtype=np.float64),
                "random": worse,
            }
        )
        self.assertEqual(friedman["n_methods"], 3)
        self.assertEqual(friedman["n_blocks"], len(better))
        self.assertEqual(len(friedman["posthoc"]), 3)
        self.assertIn("rcmoe", friedman["mean_ranks"])

        bayes = bayesian_effect_summary(better, worse, n_resamples=2_000, seed=123)
        self.assertGreater(bayes["mean_diff"], 0.0)
        self.assertGreater(bayes["prob_left_gt_right"], 0.95)
        self.assertGreater(bayes["credible_interval"][1], bayes["credible_interval"][0])


if __name__ == "__main__":
    unittest.main()
