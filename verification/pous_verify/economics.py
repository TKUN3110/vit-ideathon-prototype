"""
Cryptoeconomic Incentives, Useful-Work Stake Dynamics, and Assurance Market Engine.
Calculates historical compute score decay, lottery selection weights,
proportional reward distribution, and risk-adjusted collateral bonds.
Zero external dependencies.
"""

import math
from typing import List, Dict, Any, Tuple


class UsefulStakeTracker:
    """
    Tracks and updates non-transferable, decaying useful-work stake earned from
    verified computation contributions.
    """

    def __init__(self, decay_rate: float = 0.95, alpha: float = 0.5, epsilon: float = 1.0):
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.epsilon = epsilon
        # Worker ID -> useful stake score U_i
        self.scores: Dict[int, float] = {}

    def record_verified_work(self, worker_id: int, verified_units: int) -> None:
        current = self.scores.get(worker_id, 0.0)
        self.scores[worker_id] = current * self.decay_rate + float(verified_units)

    def apply_idle_decay(self, worker_id: int) -> None:
        current = self.scores.get(worker_id, 0.0)
        self.scores[worker_id] = current * self.decay_rate

    def penalize_worker(self, worker_id: int, penalty_fraction: float = 0.5) -> None:
        current = self.scores.get(worker_id, 0.0)
        self.scores[worker_id] = max(0.0, current * (1.0 - penalty_fraction))

    def get_selection_probabilities(self) -> Dict[int, float]:
        """
        Computes lottery selection probabilities:
        P(i) = (U_i + eps)^alpha / sum((U_j + eps)^alpha)
        """
        if not self.scores:
            return {}
        raw_weights = {
            wid: (score + self.epsilon) ** self.alpha
            for wid, score in self.scores.items()
        }
        total_weight = sum(raw_weights.values())
        if total_weight == 0.0:
            equal_prob = 1.0 / len(self.scores)
            return {wid: equal_prob for wid in self.scores}
        return {wid: w / total_weight for wid, w in raw_weights.items()}


class AssuranceTierCalculator:
    """
    Formalizes the Assurance Market:
    Maps desired security parameters to required audit queries, verification fees,
    and required worker collateral bonds.
    """

    @staticmethod
    def calculate_required_queries(
        security_bits: int,
        effective_distance: float
    ) -> int:
        """
        k >= (lambda * ln(2)) / ln(1 / (1 - delta))
        """
        if effective_distance <= 0.0 or effective_distance >= 1.0:
            raise ValueError("Effective distance must be in (0, 1).")
        numerator = security_bits * math.log(2.0)
        denominator = math.log(1.0 / (1.0 - effective_distance))
        return int(math.ceil(numerator / denominator))

    @staticmethod
    def calculate_tier_specs(
        tier_name: str,
        security_bits: int,
        effective_distance: float,
        job_value: float,
        cost_per_query: float = 0.5,
        base_fee: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Computes comprehensive tier parameters:
        queries, detection probability, protocol fee, and required bond.
        """
        queries = AssuranceTierCalculator.calculate_required_queries(security_bits, effective_distance)
        miss_prob = (1.0 - effective_distance) ** queries
        p_detect = 1.0 - miss_prob
        fee = base_fee + queries * cost_per_query

        # Rational cheater condition: (1 - P_detect) * JobValue <= P_detect * Bond
        # Minimum Bond = JobValue * (1 - P_detect) / P_detect
        # Safe bonded ratio:
        min_bond = job_value if p_detect >= 0.5 else job_value / max(1e-9, p_detect)

        return {
            "tier_name": tier_name,
            "security_bits": security_bits,
            "effective_distance": effective_distance,
            "required_queries": queries,
            "detection_probability": p_detect,
            "miss_probability": miss_prob,
            "verification_fee": fee,
            "min_worker_bond": round(min_bond, 2),
        }


def distribute_proportional_rewards(
    accepted_work: Dict[int, int],
    total_reward_pool: float
) -> Dict[int, float]:
    """
    Distributes task reward pool strictly in proportion to accepted unique canonical units.
    """
    total_units = sum(accepted_work.values())
    if total_units == 0:
        return {wid: 0.0 for wid in accepted_work}
    return {
        wid: (units / total_units) * total_reward_pool
        for wid, units in accepted_work.items()
    }
