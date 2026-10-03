"""
Unit tests for mathematical models, hypergeometric sampling, and economics.
"""

import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pous_verify.simulator import (
    exact_hypergeometric_detection,
    wilson_score_interval,
    simulate_sampling_detection,
    committee_byzantine_capture_prob,
)
from pous_verify.economics import (
    UsefulStakeTracker,
    AssuranceTierCalculator,
    distribute_proportional_rewards,
)


class TestMathAndEconomics(unittest.TestCase):
    def test_hypergeometric_exact_boundary(self):
        # 100 total, 10 corrupt, 0 samples -> 0.0
        self.assertEqual(exact_hypergeometric_detection(100, 10, 0), 0.0)
        # 100 total, 0 corrupt, 10 samples -> 0.0
        self.assertEqual(exact_hypergeometric_detection(100, 0, 10), 0.0)
        # 100 total, 100 corrupt, 1 sample -> 1.0
        self.assertEqual(exact_hypergeometric_detection(100, 100, 1), 1.0)
        # Sampling all remaining non-corrupt + 1 guarantees detection:
        # N=100, c=10. Non-corrupt=90. k=91 -> 1.0
        self.assertEqual(exact_hypergeometric_detection(100, 10, 91), 1.0)

    def test_monte_carlo_vs_exact_hypergeometric(self):
        res = simulate_sampling_detection(
            n_total=1000,
            c_corrupt=100,  # 10% corrupt
            k_samples=30,
            trials=3000,
            seed=42
        )
        # Empirical proportion must be within 95% Wilson confidence interval
        self.assertGreaterEqual(res["empirical_p"], res["ci_lower_95"])
        self.assertLessEqual(res["empirical_p"], res["ci_upper_95"])
        # Absolute error between empirical and exact must be small (< 0.02 for 3000 trials)
        self.assertLess(res["abs_error"], 0.02)

    def test_byzantine_committee_capture_prob(self):
        # Total validators = 100, Byzantine = 20 (20%).
        # Committee size = 7, Quorum = 5.
        p_cap = committee_byzantine_capture_prob(
            total_validators=100,
            byzantine_validators=20,
            committee_size=7,
            quorum_size=5
        )
        # Expected capture probability for 20% Byzantine with quorum 5/7 is very low (< 1%)
        self.assertLess(p_cap, 0.01)
        self.assertGreater(p_cap, 0.0)

    def test_useful_stake_decay_and_proportional_rewards(self):
        tracker = UsefulStakeTracker(decay_rate=0.9, alpha=0.5, epsilon=0.1)
        tracker.record_verified_work(worker_id=1, verified_units=100)
        tracker.record_verified_work(worker_id=2, verified_units=400)

        probs = tracker.get_selection_probabilities()
        # Worker 2 did 4x work, but with sqrt alpha=0.5, selection weight is ~ 2x Worker 1
        self.assertGreater(probs[2], probs[1])
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

        # Proportional rewards (linear with accepted units):
        rewards = distribute_proportional_rewards({1: 100, 2: 400}, total_reward_pool=1000.0)
        self.assertAlmostEqual(rewards[1], 200.0)
        self.assertAlmostEqual(rewards[2], 800.0)

    def test_assurance_tier_calculator(self):
        # Security: 40 bits (~10^-12), distance = 0.75 (blowup factor 4)
        queries = AssuranceTierCalculator.calculate_required_queries(security_bits=40, effective_distance=0.75)
        # 40 * ln(2) / ln(4) = 40 / 2 = 20 queries!
        self.assertEqual(queries, 20)

        tier_info = AssuranceTierCalculator.calculate_tier_specs(
            tier_name="Standard-Coded",
            security_bits=40,
            effective_distance=0.75,
            job_value=5000.0
        )
        self.assertGreaterEqual(tier_info["detection_probability"], 1.0 - 1e-12)
        self.assertEqual(tier_info["required_queries"], 20)


if __name__ == "__main__":
    unittest.main()
