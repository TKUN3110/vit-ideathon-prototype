"""
Deterministic State-Transition Engine for Proof of Useful Stake.
Implements canonical fixed-point (Q16.16) arithmetic and neural-update operations
guaranteeing 100% bit-exact reproducibility across architectures.
"""

import struct
from typing import List, Tuple, Dict, Any
from .crypto import sha256


# Fixed-point scale factor: 16 fractional bits (1.0 = 65536)
SCALE = 65536
HALF = SCALE // 2


def float_to_fixed(val: float) -> int:
    """Convert float to signed 32-bit fixed point Q16.16."""
    return int(round(val * SCALE))


def fixed_to_float(val: int) -> float:
    """Convert signed 32-bit fixed point Q16.16 to float."""
    return val / SCALE


def fixed_mul(a: int, b: int) -> int:
    """Multiply two Q16.16 integers with rounding."""
    product = a * b
    if product >= 0:
        return (product + HALF) // SCALE
    else:
        return (product - HALF) // SCALE


class ModelWeights:
    """
    Fixed-point quantized model weights and biases for a d-dimensional model.
    """

    def __init__(self, weights: List[int], bias: int):
        self.weights = list(weights)
        self.bias = bias

    def serialize(self) -> bytes:
        """Binary packed representation for deterministic hashing."""
        header = struct.pack(">I", len(self.weights))
        w_bytes = struct.pack(f">{len(self.weights)}i", *self.weights)
        b_bytes = struct.pack(">i", self.bias)
        return header + w_bytes + b_bytes

    @classmethod
    def deserialize(cls, data: bytes) -> "ModelWeights":
        (length,) = struct.unpack_from(">I", data, 0)
        offset = 4
        weights = struct.unpack_from(f">{length}i", data, offset)
        offset += length * 4
        (bias,) = struct.unpack_from(">i", data, offset)
        return cls(list(weights), bias)

    def state_hash(self) -> bytes:
        return sha256(self.serialize())


class Minibatch:
    """
    Minibatch consisting of integer-quantized feature vectors and target labels.
    """

    def __init__(self, x_samples: List[List[int]], y_samples: List[int]):
        self.x = [list(sample) for sample in x_samples]
        self.y = list(y_samples)

    def serialize(self) -> bytes:
        n_samples = len(self.y)
        d_dim = len(self.x[0]) if n_samples > 0 else 0
        header = struct.pack(">II", n_samples, d_dim)
        data = bytearray(header)
        for row in self.x:
            data.extend(struct.pack(f">{d_dim}i", *row))
        data.extend(struct.pack(f">{n_samples}i", *self.y))
        return bytes(data)

    def batch_hash(self) -> bytes:
        return sha256(self.serialize())


def deterministic_train_step(
    current_state: ModelWeights,
    batch: Minibatch,
    learning_rate: int = float_to_fixed(0.01)
) -> ModelWeights:
    """
    Executes one deterministic gradient-descent transition:
    S_{t+1} = Step(S_t, D_t, eta)
    Computes predictions, residuals, gradients in fixed-point, and updates weights.
    """
    n_samples = len(batch.y)
    if n_samples == 0:
        return ModelWeights(current_state.weights, current_state.bias)

    d_dim = len(current_state.weights)
    grad_w = [0] * d_dim
    grad_b = 0

    for i in range(n_samples):
        xi = batch.x[i]
        yi = batch.y[i]

        # Forward pass: y_hat = sum(w_j * x_j) + b
        y_hat = current_state.bias
        for j in range(d_dim):
            y_hat += fixed_mul(current_state.weights[j], xi[j])

        # Residual: error = y_hat - yi
        residual = y_hat - yi

        # Gradient accumulation
        for j in range(d_dim):
            grad_w[j] += fixed_mul(residual, xi[j])
        grad_b += residual

    # Average gradients
    for j in range(d_dim):
        grad_w[j] //= n_samples
    grad_b //= n_samples

    # Update: W_{t+1} = W_t - eta * grad_w
    new_weights = [0] * d_dim
    for j in range(d_dim):
        delta = fixed_mul(learning_rate, grad_w[j])
        new_weights[j] = current_state.weights[j] - delta
    delta_b = fixed_mul(learning_rate, grad_b)
    new_bias = current_state.bias - delta_b

    return ModelWeights(new_weights, new_bias)


class ExecutionTrace:
    """
    Manages a deterministic trajectory of T sequential training steps,
    recording state hashes, transition leaves, and Merkleized trace.
    """

    def __init__(self, task_id: bytes, initial_state: ModelWeights):
        self.task_id = task_id
        self.states: List[ModelWeights] = [initial_state]
        self.batches: List[Minibatch] = []
        self.leaves: List[bytes] = []

    def append_step(self, batch: Minibatch, learning_rate: int = float_to_fixed(0.01)) -> None:
        t = len(self.batches)
        current_state = self.states[-1]
        next_state = deterministic_train_step(current_state, batch, learning_rate)

        # Transition leaf: H(task_id || t || H(S_t) || H(D_t) || H(S_{t+1}))
        leaf_payload = (
            self.task_id
            + struct.pack(">I", t)
            + current_state.state_hash()
            + batch.batch_hash()
            + next_state.state_hash()
        )
        self.leaves.append(sha256(leaf_payload))
        self.batches.append(batch)
        self.states.append(next_state)

    @property
    def total_steps(self) -> int:
        return len(self.batches)

    @property
    def final_state(self) -> ModelWeights:
        return self.states[-1]
