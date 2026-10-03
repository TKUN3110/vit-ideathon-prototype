"""
Standalone CLI script to execute FedRisk Flower Simulation.
Multiplexes RTX 5060 across 3 simulated hospital nodes with SMPC Secure Aggregation.
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fedrisk.config import NUM_FEDERATED_ROUNDS
from fedrisk.federated.simulation import run_federated_simulation


def main():
    parser = argparse.ArgumentParser(description="FedRisk Flower Simulation Engine Runner")
    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_FEDERATED_ROUNDS,
        help="Number of federated aggregation rounds (default: 5)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU execution (disable RTX 5060)",
    )
    args = parser.parse_args()

    print("=" * 75)
    print("[FedRisk] Privacy-Preserving Federated Simulation")
    print(f"Rounds: {args.rounds} | Nodes: 3 | SMPC: Enabled | Device: {'CPU' if args.no_gpu else 'RTX 5060 (0.33/node)'}")
    print("=" * 75)

    result = run_federated_simulation(
        num_rounds=args.rounds,
        use_gpu=(not args.no_gpu),
    )

    print("\n" + "=" * 75)
    print(f"Simulation Finished! Status: {result.get('status')}")
    print(f"Best Validation AUROC Achieved: {result.get('best_auroc', 0.0):.4f}")
    print(f"SMPC Zero-Sum Mask Verifications: {len(result.get('smpc_audits', []))} rounds verified.")
    print("Checkpoints saved to ./checkpoints/")
    print("=" * 75)


if __name__ == "__main__":
    main()
