"""
Unit tests for Reed-Solomon Low-Degree Extension (LDE) and error amplification.
"""

import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pous_verify.coding import LowDegreeExtension, sample_query_detection_prob


class TestCodingAmplification(unittest.TestCase):
    def test_exact_interpolation_on_raw_domain(self):
        raw_trace = [12, 45, 99, 1024, 7, 88, 300, 512]
        lde = LowDegreeExtension(raw_trace, blowup_factor=4)

        # First N elements of codeword must exactly match the raw trace
        for i, val in enumerate(raw_trace):
            self.assertEqual(lde.codeword[i], val)

    def test_single_error_amplification_theorem(self):
        """
        Theorem: Altering even 1 element in raw trace of length N guarantees that
        two distinct degree-(N-1) polynomials differ in at least M - (N - 1) positions.
        With blowup B=4, code rate R=0.25, relative distance delta >= 1 - R = 0.75.
        """
        raw_honest = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
        lde_honest = LowDegreeExtension(raw_honest, blowup_factor=4)

        # Adversary modifies just 1 element (sparse fraud at index 5)
        raw_fraudulent = list(raw_honest)
        raw_fraudulent[5] = 9999
        lde_fraudulent = LowDegreeExtension(raw_fraudulent, blowup_factor=4)

        diff_count, rel_distance = lde_honest.compare_codeword(lde_fraudulent.codeword)

        # N=16, M=64. Minimum differing positions: 64 - 15 = 49 (rel distance = 49/64 = 0.7656)
        self.assertGreaterEqual(diff_count, 49)
        self.assertGreaterEqual(rel_distance, 0.75)

    def test_query_soundness_calculation(self):
        distance = 0.75
        # 1 query: 75% detection
        self.assertAlmostEqual(sample_query_detection_prob(distance, 1), 0.75, places=4)
        # 10 queries: 1 - 0.25^10 >= 0.999999
        self.assertGreaterEqual(sample_query_detection_prob(distance, 10), 0.99999)
        # 20 queries: ~ 1.0
        self.assertGreaterEqual(sample_query_detection_prob(distance, 20), 0.999999999)


if __name__ == "__main__":
    unittest.main()
