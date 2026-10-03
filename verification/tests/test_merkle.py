"""
Unit tests for domain-separated Merkle trees and cryptographic proofs.
"""

import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pous_verify.crypto import MerkleTree, hash_leaf, hash_internal, sha256


class TestMerkleTree(unittest.TestCase):
    def test_single_leaf(self):
        leaf = b"test_leaf_0"
        tree = MerkleTree([leaf])
        self.assertEqual(tree.root, hash_leaf(leaf))
        proof = tree.get_proof(0)
        self.assertEqual(proof, [("right", hash_leaf(leaf))])
        self.assertTrue(MerkleTree.verify_proof(tree.root, leaf, proof))

    def test_multiple_leaves_inclusion(self):
        leaves = [f"transition_{i}".encode("utf-8") for i in range(16)]
        tree = MerkleTree(leaves)
        self.assertEqual(len(tree.root), 32)

        for i, leaf in enumerate(leaves):
            proof = tree.get_proof(i)
            # Proof length for 16 leaves must be exactly 4 (log2(16))
            self.assertEqual(len(proof), 4)
            self.assertTrue(MerkleTree.verify_proof(tree.root, leaf, proof))

    def test_tamper_leaf_rejection(self):
        leaves = [f"state_{i}".encode("utf-8") for i in range(8)]
        tree = MerkleTree(leaves)

        proof_0 = tree.get_proof(0)
        # Attempt to verify modified leaf data against genuine proof
        fake_leaf = b"state_0_tampered"
        self.assertFalse(MerkleTree.verify_proof(tree.root, fake_leaf, proof_0))

    def test_tamper_proof_rejection(self):
        leaves = [f"step_{i}".encode("utf-8") for i in range(8)]
        tree = MerkleTree(leaves)

        proof_3 = tree.get_proof(3)
        # Corrupt one sibling hash in the proof
        corrupted_proof = list(proof_3)
        direction, sib_hash = corrupted_proof[0]
        tampered_sib = bytearray(sib_hash)
        tampered_sib[0] ^= 0xFF
        corrupted_proof[0] = (direction, bytes(tampered_sib))

        self.assertFalse(MerkleTree.verify_proof(tree.root, leaves[3], corrupted_proof))


if __name__ == "__main__":
    unittest.main()
