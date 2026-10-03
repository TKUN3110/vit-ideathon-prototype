"""
Domain-Separated Cryptographic Primitives for PoUS.
Implements SHA-256 Merkle Trees with leaf (0x00) and interior (0x01) prefixes,
inclusion proof generation, and verification routines.
Zero external dependencies.
"""

import hashlib
from typing import List, Tuple, Optional


def sha256(data: bytes) -> bytes:
    """Standard SHA-256 digest."""
    return hashlib.sha256(data).digest()


def hash_leaf(data: bytes) -> bytes:
    """Domain-separated leaf hash: H(0x00 || data)"""
    return sha256(b"\x00" + data)


def hash_internal(left: bytes, right: bytes) -> bytes:
    """Domain-separated internal node hash: H(0x01 || left || right)"""
    return sha256(b"\x01" + left + right)


class MerkleTree:
    """
    Binary Merkle Tree supporting domain-separated hashing,
    audit proof generation, and proof verification.
    """

    def __init__(self, leaves_data: List[bytes]):
        if not leaves_data:
            raise ValueError("Cannot construct MerkleTree with empty leaves.")
        self.raw_leaves = list(leaves_data)
        self.leaf_hashes: List[bytes] = [hash_leaf(leaf) for leaf in self.raw_leaves]
        self.levels: List[List[bytes]] = [self.leaf_hashes]
        self._build_tree()

    def _build_tree(self) -> None:
        current_level = self.leaf_hashes
        while len(current_level) > 1:
            next_level: List[bytes] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    # Canonical duplicate for odd node count
                    right = left
                next_level.append(hash_internal(left, right))
            self.levels.append(next_level)
            current_level = next_level

    @property
    def root(self) -> bytes:
        """Root hash of the Merkle Tree (32 bytes)."""
        return self.levels[-1][0]

    @property
    def root_hex(self) -> str:
        """Hex-encoded root hash."""
        return self.root.hex()

    def get_proof(self, index: int) -> List[Tuple[str, bytes]]:
        """
        Generate Merkle audit path for leaf at index.
        Returns list of tuples: (position ('left' | 'right'), sibling_hash).
        """
        if index < 0 or index >= len(self.raw_leaves):
            raise IndexError(f"Leaf index {index} out of range (0..{len(self.raw_leaves)-1})")

        proof: List[Tuple[str, bytes]] = []
        idx = index
        for level in self.levels[:-1]:
            if idx % 2 == 0:
                # Target is left child, sibling is right
                sibling_idx = idx + 1 if idx + 1 < len(level) else idx
                proof.append(("right", level[sibling_idx]))
            else:
                # Target is right child, sibling is left
                sibling_idx = idx - 1
                proof.append(("left", level[sibling_idx]))
            idx //= 2
        return proof

    @staticmethod
    def verify_proof(root: bytes, leaf_data: bytes, proof: List[Tuple[str, bytes]]) -> bool:
        """
        Verify that leaf_data is part of the Merkle tree with given root.
        """
        current_hash = hash_leaf(leaf_data)
        for direction, sibling_hash in proof:
            if direction == "right":
                current_hash = hash_internal(current_hash, sibling_hash)
            elif direction == "left":
                current_hash = hash_internal(sibling_hash, current_hash)
            else:
                return False
        return current_hash == root
