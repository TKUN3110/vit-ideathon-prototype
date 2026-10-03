"""
Unit tests for interactive Merkle trace bisection dispute resolution.
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
)
from pous_verify.dispute import DisputeSession


class TestDisputeBisection(unittest.TestCase):
    def setUp(self):
        task_id = b"dispute_task_001"
        initial_state = ModelWeights([float_to_fixed(1.0), float_to_fixed(-0.5)], float_to_fixed(0.2))

        # Generate 64 deterministic batches
        self.batches = []
        for i in range(64):
            self.batches.append(
                Minibatch(
                    x_samples=[[float_to_fixed(float(i % 5)), float_to_fixed(float((i + 1) % 3))]],
                    y_samples=[float_to_fixed(float(i % 2))],
                )
            )

        # Worker A computes 64 honest steps
        trace_a = ExecutionTrace(task_id, initial_state)
        for b in self.batches:
            trace_a.append_step(b)
        self.states_a = trace_a.states

        # Worker B computes honest steps up to step 41, but at step 42 injects fraud
        self.states_b = [initial_state]
        for idx, b in enumerate(self.batches):
            current = self.states_b[-1]
            next_state = deterministic_train_step(current, b)
            if idx == 42:
                # Malicious weight perturbation
                fraudulent_weights = list(next_state.weights)
                fraudulent_weights[0] += float_to_fixed(5.0)
                next_state = ModelWeights(fraudulent_weights, next_state.bias)
            self.states_b.append(next_state)

    def test_bisection_pinpoints_exact_fraud_step(self):
        session = DisputeSession(
            dispute_id="dsp_001",
            worker_a=1,
            worker_b=2,
            trace_a_states=self.states_a,
            trace_b_states=self.states_b,
            batches=self.batches,
            bond_a=1000,
            bond_b=1000,
        )

        receipt = session.resolve_dispute()

        # In 64 steps, ceil(log2(64)) = 6 bisection rounds
        self.assertEqual(receipt["bisection_rounds_count"], 6)
        # Disputed step must be exactly index 42
        self.assertEqual(receipt["disputed_step"], 42)
        # Decision must slash Worker B
        self.assertEqual(receipt["decision"], "WORKER_B_FRAUD_SLASHED")
        self.assertEqual(receipt["guilty_worker"], 2)
        self.assertEqual(receipt["innocent_worker"], 1)
        self.assertEqual(receipt["slashed_amount"], 1000)
        self.assertEqual(receipt["challenger_bounty"], 500)


if __name__ == "__main__":
    unittest.main()
