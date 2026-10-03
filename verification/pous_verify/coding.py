"""
Error-Amplification Engine for Proof of Useful Stake.
Implements Reed-Solomon Low-Degree Extension (LDE) over prime field F_p,
demonstrating PCP-style gap amplification where sparse single-step errors
spread across >= (1 - R) fraction of the encoded domain.
Zero external dependencies.
"""

from typing import List, Tuple
import math


# Prime field modulus: Fermat prime 2^16 + 1 = 65537
FIELD_PRIME = 65537


def add_mod(a: int, b: int, p: int = FIELD_PRIME) -> int:
    return (a + b) % p


def sub_mod(a: int, b: int, p: int = FIELD_PRIME) -> int:
    return (a - b + p) % p


def mul_mod(a: int, b: int, p: int = FIELD_PRIME) -> int:
    return (a * b) % p


def inv_mod(a: int, p: int = FIELD_PRIME) -> int:
    """Extended Euclidean algorithm for modular inverse."""
    a = a % p
    if a == 0:
        raise ZeroDivisionError("Cannot invert 0 in finite field.")
    # Fermat's Little Theorem: a^(p-2) mod p for prime p
    return pow(a, p - 2, p)


class LowDegreeExtension:
    """
    Encodes raw computational traces into low-degree polynomial evaluations
    over a larger domain, providing error amplification.
    """

    def __init__(self, trace: List[int], blowup_factor: int = 4, p: int = FIELD_PRIME):
        if not trace:
            raise ValueError("Trace cannot be empty.")
        if blowup_factor < 2:
            raise ValueError("Blowup factor must be at least 2.")
        self.p = p
        self.n = len(trace)
        self.blowup = blowup_factor
        self.m = self.n * self.blowup
        if self.m >= self.p:
            raise ValueError(f"Extended domain size {self.m} exceeds field size {self.p}.")

        self.raw_trace = [val % self.p for val in trace]
        self.code_rate = self.n / self.m
        self.theoretical_min_distance = 1.0 - self.code_rate

        # Interpolate polynomial coefficients in monomial basis or evaluate via Lagrange
        self.codeword = self._compute_lde()

    def _compute_lde(self) -> List[int]:
        """
        Evaluates the unique degree-(n-1) polynomial interpolating
        (0, raw[0]), (1, raw[1]), ..., (n-1, raw[n-1])
        at all points x in 0..m-1 using barycentric/Lagrange evaluation.
        """
        # Precompute weights for domain points 0..n-1
        # w_j = 1 / prod_{k != j} (j - k)
        weights = [1] * self.n
        for j in range(self.n):
            denom = 1
            for k in range(self.n):
                if k != j:
                    diff = sub_mod(j, k, self.p)
                    denom = mul_mod(denom, diff, self.p)
            weights[j] = inv_mod(denom, self.p)

        codeword = [0] * self.m

        # For x in 0..n-1, value is exactly the raw trace
        for x in range(self.n):
            codeword[x] = self.raw_trace[x]

        # For x in n..m-1, use Lagrange formula
        for x in range(self.n, self.m):
            num = 0
            den = 0
            for j in range(self.n):
                diff = sub_mod(x, j, self.p)
                inv_diff = inv_mod(diff, self.p)
                term = mul_mod(weights[j], inv_diff, self.p)
                num = add_mod(num, mul_mod(term, self.raw_trace[j], self.p), self.p)
                den = add_mod(den, term, self.p)
            codeword[x] = mul_mod(num, inv_mod(den, self.p), self.p)

        return codeword

    def compare_codeword(self, other_codeword: List[int]) -> Tuple[int, float]:
        """
        Compares this codeword against another.
        Returns: (differing_positions_count, relative_hamming_distance).
        """
        if len(self.codeword) != len(other_codeword):
            raise ValueError("Codewords must have identical length.")
        diff_count = sum(1 for a, b in zip(self.codeword, other_codeword) if a != b)
        relative_distance = diff_count / self.m
        return diff_count, relative_distance


def sample_query_detection_prob(distance: float, queries: int) -> float:
    """
    Analytical detection probability given relative distance delta and q queries:
    P_detect >= 1 - (1 - delta)^q
    """
    if queries <= 0:
        return 0.0
    miss_prob = (1.0 - distance) ** queries
    return 1.0 - miss_prob
