"""
Interactive Merkle Trace Bisection and Dispute Adjudication Engine.
Executes logarithmic bisection over divergent execution traces down to a single step,
deterministically resolves the valid state transition, and issues objective slashing receipts.
Zero external dependencies.
"""

from typing import List, Dict, Tuple, Any, Optional
from .crypto import sha256
from .deterministic import ModelWeights, Minibatch, deterministic_train_step


class DisputeSession:
    """
    Manages an interactive bisection dispute between Worker A and Worker B
    over an overlapping training interval of length T.
    """

    def __init__(
        self,
        dispute_id: str,
        worker_a: int,
        worker_b: int,
        trace_a_states: List[ModelWeights],
        trace_b_states: List[ModelWeights],
        batches: List[Minibatch],
        bond_a: int,
        bond_b: int,
    ):
        if len(trace_a_states) != len(trace_b_states):
            raise ValueError("Traces must have identical length for bisection.")
        if len(batches) != len(trace_a_states) - 1:
            raise ValueError("Batches count must equal states count minus 1.")

        self.dispute_id = dispute_id
        self.worker_a = worker_a
        self.worker_b = worker_b
        self.trace_a = trace_a_states
        self.trace_b = trace_b_states
        self.batches = batches
        self.bond_a = bond_a
        self.bond_b = bond_b

        self.total_steps = len(batches)
        self.bisection_rounds: List[Dict[str, Any]] = []

    def execute_bisection(self) -> Tuple[int, ModelWeights, Minibatch, ModelWeights, ModelWeights]:
        """
        Executes interactive binary search in ceil(log2(T)) rounds.
        Returns:
            (disputed_step_t, input_state, batch, claimed_next_a, claimed_next_b)
        """
        low = 0
        high = self.total_steps

        # Check precondition: initial states must agree
        if self.trace_a[low].state_hash() != self.trace_b[low].state_hash():
            raise RuntimeError("Initial states differ before training began.")

        # Check precondition: final states must diverge
        if self.trace_a[high].state_hash() == self.trace_b[high].state_hash():
            raise RuntimeError("Final states match; no dispute exists.")

        round_num = 0
        while high - low > 1:
            mid = (low + high) // 2
            hash_a = self.trace_a[mid].state_hash()
            hash_b = self.trace_b[mid].state_hash()
            matches = (hash_a == hash_b)

            self.bisection_rounds.append({
                "round": round_num,
                "low": low,
                "mid": mid,
                "high": high,
                "hash_a": hash_a.hex()[:16],
                "hash_b": hash_b.hex()[:16],
                "matches": matches,
            })

            if matches:
                # Discrepancy happened after mid
                low = mid
            else:
                # Discrepancy happened before mid
                high = mid
            round_num += 1

        # At this point, high - low == 1
        t_star = low
        input_state = self.trace_a[t_star]
        batch = self.batches[t_star]
        claimed_next_a = self.trace_a[t_star + 1]
        claimed_next_b = self.trace_b[t_star + 1]

        return t_star, input_state, batch, claimed_next_a, claimed_next_b

    def resolve_dispute(self) -> Dict[str, Any]:
        """
        Locates the exact single disputed step via bisection,
        evaluates the step deterministically, and issues an objective slashing receipt.
        """
        t_star, input_state, batch, next_a, next_b = self.execute_bisection()

        # Deterministic single-step recomputation
        canonical_next = deterministic_train_step(input_state, batch)
        canonical_hash = canonical_next.state_hash()

        hash_a = next_a.state_hash()
        hash_b = next_b.state_hash()

        a_correct = (hash_a == canonical_hash)
        b_correct = (hash_b == canonical_hash)

        if a_correct and not b_correct:
            guilty = self.worker_b
            innocent = self.worker_a
            slashed_amount = self.bond_b
            bounty = slashed_amount // 2
            decision = "WORKER_B_FRAUD_SLASHED"
        elif b_correct and not a_correct:
            guilty = self.worker_a
            innocent = self.worker_b
            slashed_amount = self.bond_a
            bounty = slashed_amount // 2
            decision = "WORKER_A_FRAUD_SLASHED"
        elif not a_correct and not b_correct:
            guilty = -1  # Both guilty
            innocent = 0
            slashed_amount = self.bond_a + self.bond_b
            bounty = 0
            decision = "BOTH_WORKERS_FRAUD_SLASHED"
        else:
            decision = "UNEXPECTED_SPEC_EQUIVOCATION"
            guilty = 0
            innocent = 0
            slashed_amount = 0
            bounty = 0

        receipt = {
            "dispute_id": self.dispute_id,
            "decision": decision,
            "total_steps": self.total_steps,
            "bisection_rounds_count": len(self.bisection_rounds),
            "disputed_step": t_star,
            "canonical_state_hash": canonical_hash.hex(),
            "guilty_worker": guilty,
            "innocent_worker": innocent,
            "slashed_amount": slashed_amount,
            "challenger_bounty": bounty,
            "rounds_detail": self.bisection_rounds,
        }

        return receipt
