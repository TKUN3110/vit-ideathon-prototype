"""
Monte Carlo Simulation and Exact Statistical Verification Engine for PoUS.
Implements exact hypergeometric detection probabilities, Wilson score 95%
confidence intervals, Byzantine committee capture analysis, and multi-node simulations.
Zero external dependencies.
"""

import math
import random
from typing import List, Dict, Tuple, Any


def comb(n: int, k: int) -> int:
    """Exact combinatorial calculation math.comb."""
    return math.comb(n, k)


def exact_hypergeometric_detection(n_total: int, c_corrupt: int, k_samples: int) -> float:
    """
    Exact probability of detecting at least 1 corrupt unit when sampling k units without replacement:
    P_detect = 1 - (comb(n - c, k) / comb(n, k))
    """
    if k_samples <= 0:
        return 0.0
    if c_corrupt <= 0:
        return 0.0
    if c_corrupt >= n_total:
        return 1.0
    if k_samples >= n_total - c_corrupt + 1:
        return 1.0

    num = comb(n_total - c_corrupt, k_samples)
    den = comb(n_total, k_samples)
    miss_prob = num / den
    return 1.0 - miss_prob


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Wilson score interval for binomial proportion.
    Default confidence 95% (z = 1.95996).
    Returns (center, lower_bound, upper_bound).
    """
    if trials == 0:
        return 0.0, 0.0, 0.0
    p = successes / trials
    # z for 95% = 1.95996
    z = 1.95996
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return p, lower, upper


def simulate_sampling_detection(
    n_total: int,
    c_corrupt: int,
    k_samples: int,
    trials: int = 5000,
    seed: int = 2601
) -> Dict[str, Any]:
    """
    Runs Monte Carlo simulation of sampling k elements from n containing c corrupt items,
    and compares against the exact analytical hypergeometric formula.
    """
    rng = random.Random(seed)
    detected_count = 0

    population = [1] * c_corrupt + [0] * (n_total - c_corrupt)

    for _ in range(trials):
        # Sample k without replacement
        sample = rng.sample(population, k_samples)
        if 1 in sample:
            detected_count += 1

    emp_p, lower, upper = wilson_score_interval(detected_count, trials)
    exact_p = exact_hypergeometric_detection(n_total, c_corrupt, k_samples)

    return {
        "n_total": n_total,
        "c_corrupt": c_corrupt,
        "k_samples": k_samples,
        "trials": trials,
        "detected_count": detected_count,
        "empirical_p": emp_p,
        "ci_lower_95": lower,
        "ci_upper_95": upper,
        "exact_p": exact_p,
        "abs_error": abs(emp_p - exact_p),
    }


def committee_byzantine_capture_prob(
    total_validators: int,
    byzantine_validators: int,
    committee_size: int,
    quorum_size: int
) -> float:
    """
    Exact probability that a randomly drawn committee of size committee_size
    contains at least quorum_size Byzantine members (committee capture):
    Sum_{j = quorum}^{committee} [comb(B, j) * comb(N - B, r - j)] / comb(N, r)
    """
    den = comb(total_validators, committee_size)
    if den == 0:
        return 0.0

    capture_favorable = 0
    max_byz = min(byzantine_validators, committee_size)
    for j in range(quorum_size, max_byz + 1):
        rem_honest = committee_size - j
        if rem_honest <= total_validators - byzantine_validators:
            capture_favorable += comb(byzantine_validators, j) * comb(total_validators - byzantine_validators, rem_honest)

    return capture_favorable / den
