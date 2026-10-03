"""
Standalone CLI script to generate synthetic FHIR patient cohorts across 3 hospital partitions.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fedrisk.data.dataset_generator import FHIRDatasetGenerator


def main():
    print("=" * 70)
    print("FedRisk: Synthetic FHIR ICU Cohort Generator")
    print("Generating partitioned non-IID datasets across 3 hospital nodes...")
    print("=" * 70)

    generator = FHIRDatasetGenerator(seed=42)
    partitions = generator.generate_partitioned_dataset()

    total_patients = sum(len(p) for p in partitions.values())
    print("\n" + "=" * 70)
    print(f"Successfully generated {total_patients} FHIR patient trajectories across 3 sites!")
    print("Data partitions saved to ./data/partitions/")
    print("=" * 70)


if __name__ == "__main__":
    main()
