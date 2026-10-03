"""
Blind Partial-Overlap Auditing (BPOA) Allocator.
Coordinates task assignment, post-commitment threshold randomness beacons,
and pair-specific overlapping audit hypergraphs.
Zero external dependencies.
"""

import struct
from typing import List, Dict, Set, Tuple, Any
from .crypto import sha256


class WorkerRegistration:
    def __init__(self, worker_id: int, declared_capacity: int, collateral_bond: int):
        self.worker_id = worker_id
        self.declared_capacity = declared_capacity
        self.collateral_bond = collateral_bond
        self.assigned_units: int = 0
        self.commitment_root: bytes = b""
        self.opened_leaves: Dict[int, bytes] = {}


class BPOACoordinator:
    """
    Coordinates the four-stage BPOA round:
    1. Allocation (capacity & collateral)
    2. Submission (commit-before-challenge)
    3. Post-Commit Beacon & Overlap Audit Generation
    4. Quorum Audit Verification
    """

    def __init__(
        self,
        task_id: bytes,
        total_work_units: int,
        collateral_ratio: float = 0.5,
        committee_size: int = 7,
        byzantine_threshold: int = 2,
    ):
        self.task_id = task_id
        self.total_work_units = total_work_units
        self.collateral_ratio = collateral_ratio
        self.committee_size = committee_size
        self.byzantine_threshold = byzantine_threshold
        self.quorum_size = 2 * byzantine_threshold + 1

        self.workers: Dict[int, WorkerRegistration] = {}
        self.beacon_seed: bytes = b""
        self.audit_hypergraph: List[Dict[str, Any]] = []

    def register_worker(self, worker_id: int, declared_capacity: int, collateral_bond: int) -> bool:
        """Register worker if collateral meets threshold."""
        min_bond = int(declared_capacity * self.collateral_ratio)
        if collateral_bond < min_bond:
            return False
        self.workers[worker_id] = WorkerRegistration(worker_id, declared_capacity, collateral_bond)
        return True

    def allocate_work(self) -> None:
        """Allocate canonical work units proportional to declared capacity."""
        if not self.workers:
            return
        total_capacity = sum(w.declared_capacity for w in self.workers.values())
        if total_capacity == 0:
            return

        for w in self.workers.values():
            w.assigned_units = max(1, int(round((w.declared_capacity / total_capacity) * self.total_work_units)))

    def submit_commitment(self, worker_id: int, root_hash: bytes) -> bool:
        """Record cryptographic root commitment prior to beacon generation."""
        if worker_id not in self.workers:
            return False
        if len(root_hash) != 32:
            return False
        self.workers[worker_id].commitment_root = root_hash
        return True

    def generate_post_commit_beacon(self, epoch: int) -> bytes:
        """
        Emulates threshold randomness beacon derived after all workers have committed.
        R_seed = H(epoch || task_id || sum(C_i))
        """
        all_committed = all(len(w.commitment_root) == 32 for w in self.workers.values())
        if not all_committed:
            raise RuntimeError("Cannot generate beacon: not all workers have submitted commitments.")

        hasher = bytearray(struct.pack(">I", epoch) + self.task_id)
        # Sort by worker ID for deterministic aggregation
        for wid in sorted(self.workers.keys()):
            hasher.extend(self.workers[wid].commitment_root)

        self.beacon_seed = sha256(bytes(hasher))
        return self.beacon_seed

    def construct_audit_hypergraph(self, audit_rate: float = 0.1) -> List[Dict[str, Any]]:
        """
        Constructs pair-specific overlapping challenge sets and independent auditor committees
        using the post-commit beacon seed.
        """
        if not self.beacon_seed:
            raise RuntimeError("Beacon seed must be generated prior to hypergraph construction.")

        worker_ids = sorted(self.workers.keys())
        n_workers = len(worker_ids)
        if n_workers < 2:
            return []

        self.audit_hypergraph = []

        for i, wid_a in enumerate(worker_ids):
            units_a = self.workers[wid_a].assigned_units
            num_audits = max(1, int(round(units_a * audit_rate)))

            for check_idx in range(num_audits):
                # PRF derivation for target unit: H(beacon_seed || wid_a || check_idx)
                prng = sha256(self.beacon_seed + struct.pack(">II", wid_a, check_idx))
                unit_offset = struct.unpack(">I", prng[:4])[0] % units_a

                # Derive overlapping peer: wid_b
                peer_idx = (i + 1 + (struct.unpack(">I", prng[4:8])[0] % (n_workers - 1))) % n_workers
                wid_b = worker_ids[peer_idx]

                # Committee selection from remaining neutral validators
                neutral_pool = [w for w in worker_ids if w not in (wid_a, wid_b)]
                committee: List[int] = []
                if len(neutral_pool) >= self.committee_size:
                    for k in range(self.committee_size):
                        sel_hash = sha256(prng + struct.pack(">I", k))
                        sel_idx = struct.unpack(">I", sel_hash[:4])[0] % len(neutral_pool)
                        committee.append(neutral_pool.pop(sel_idx))
                else:
                    committee = list(neutral_pool)

                self.audit_hypergraph.append({
                    "audit_id": f"{wid_a}:{check_idx}",
                    "worker_a": wid_a,
                    "worker_b": wid_b,
                    "unit_offset": unit_offset,
                    "committee": committee,
                    "quorum": self.quorum_size,
                })

        return self.audit_hypergraph
