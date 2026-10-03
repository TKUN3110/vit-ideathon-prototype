"""
Unit tests for bit-exact deterministic training transitions.
"""

import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pous_verify.deterministic import (
    ModelWeights,
    Minibatch,
    deterministic_train_step,
    ExecutionTrace,
    float_to_fixed,
    fixed_to_float,
    fixed_mul,
)


class TestDeterministicEngine(unittest.TestCase):
    def test_fixed_point_arithmetic(self):
        a = float_to_fixed(1.5)
        b = float_to_fixed(2.0)
        prod = fixed_mul(a, b)
        self.assertAlmostEqual(fixed_to_float(prod), 3.0, places=4)

        c = float_to_fixed(-0.5)
        prod_neg = fixed_mul(a, c)
        self.assertAlmostEqual(fixed_to_float(prod_neg), -0.75, places=4)

    def test_bit_exact_reproducibility(self):
        # Two completely independent runs with same initial parameters and batches
        w0_a = ModelWeights([float_to_fixed(0.5), float_to_fixed(-1.2)], float_to_fixed(0.1))
        w0_b = ModelWeights([float_to_fixed(0.5), float_to_fixed(-1.2)], float_to_fixed(0.1))

        batch = Minibatch(
            x_samples=[
                [float_to_fixed(1.0), float_to_fixed(0.5)],
                [float_to_fixed(-0.5), float_to_fixed(2.0)],
            ],
            y_samples=[float_to_fixed(1.0), float_to_fixed(-1.0)],
        )

        w1_a = deterministic_train_step(w0_a, batch)
        w1_b = deterministic_train_step(w0_b, batch)

        # Hashes and weights must be 100% bit-for-bit identical
        self.assertEqual(w1_a.state_hash(), w1_b.state_hash())
        self.assertEqual(w1_a.weights, w1_b.weights)
        self.assertEqual(w1_a.bias, w1_b.bias)

    def test_execution_trace_generation(self):
        task_id = b"task_bck26_pous"
        w0 = ModelWeights([float_to_fixed(0.1)], float_to_fixed(0.0))
        trace = ExecutionTrace(task_id, w0)

        for i in range(10):
            batch = Minibatch(
                x_samples=[[float_to_fixed(float(i))]],
                y_samples=[float_to_fixed(float(i * 2))],
            )
            trace.append_step(batch)

        self.assertEqual(trace.total_steps, 10)
        self.assertEqual(len(trace.states), 11)
        self.assertEqual(len(trace.leaves), 10)


if __name__ == "__main__":
    unittest.main()
