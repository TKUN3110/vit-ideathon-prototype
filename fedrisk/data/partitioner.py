"""
Hospital Dataset Partitioner and PyG DataLoader Factory.
Loads partitioned clinical event graphs for each hospital node and prepares PyG DataLoaders.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..config import BATCH_SIZE, HOSPITAL_METADATA, PARTITIONS_DIR
from .dataset_generator import FHIRDatasetGenerator


class HospitalDataPartitioner:
    """
    Manages partition retrieval and DataLoader generation for federated hospital nodes.
    """

    def __init__(self, partitions_dir: Path = PARTITIONS_DIR, batch_size: int = BATCH_SIZE):
        self.partitions_dir = partitions_dir
        self.batch_size = batch_size
        self._ensure_data_exists()

    def _ensure_data_exists(self) -> None:
        """Verifies that all 3 hospital site datasets exist, or generates them automatically."""
        all_exist = all(
            (self.partitions_dir / f"site_{site_id}" / "graphs.pt").exists()
            for site_id in HOSPITAL_METADATA.keys()
        )
        if not all_exist:
            print("[Partitioner] Partitions missing. Generating synthetic FHIR cohorts...")
            generator = FHIRDatasetGenerator()
            generator.generate_partitioned_dataset()

    def load_site_graphs(self, site_id: int) -> List[Data]:
        """Loads all PyG patient graphs for a specific hospital node."""
        graph_file = self.partitions_dir / f"site_{site_id}" / "graphs.pt"
        if not graph_file.exists():
            raise FileNotFoundError(f"Partition graph file not found: {graph_file}")
        
        # Load with weights_only=False since PyG Data contains complex objects
        try:
            graphs = torch.load(graph_file, weights_only=False)
        except TypeError:
            graphs = torch.load(graph_file)
        return graphs

    def get_site_dataloaders(
        self,
        site_id: int,
        val_split: float = 0.2,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """
        Splits local site graphs into Train / Validation sets and creates PyG DataLoaders.
        """
        graphs = self.load_site_graphs(site_id)
        num_total = len(graphs)

        # Deterministic stratified split by readmission label
        import random
        rng = random.Random(seed + site_id)
        pos_graphs = [g for g in graphs if g.y.item() == 1.0]
        neg_graphs = [g for g in graphs if g.y.item() == 0.0]
        rng.shuffle(pos_graphs)
        rng.shuffle(neg_graphs)

        # Allocate validation samples while guaranteeing training set has both classes
        if len(pos_graphs) > 1:
            n_val_pos = max(1, int(len(pos_graphs) * val_split))
            n_val_pos = min(n_val_pos, len(pos_graphs) - 1)  # Guarantee >= 1 in train
        else:
            n_val_pos = 0  # Keep solitary positive in training set

        if len(neg_graphs) > 1:
            n_val_neg = max(1, int(len(neg_graphs) * val_split))
            n_val_neg = min(n_val_neg, len(neg_graphs) - 1)  # Guarantee >= 1 in train
        else:
            n_val_neg = 0

        val_graphs = pos_graphs[:n_val_pos] + neg_graphs[:n_val_neg]
        train_graphs = pos_graphs[n_val_pos:] + neg_graphs[n_val_neg:]


        # Create PyG DataLoaders
        train_loader = DataLoader(
            train_graphs,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_graphs,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        stats = {
            "total_samples": num_total,
            "train_samples": len(train_graphs),
            "val_samples": len(val_graphs),
            "train_pos": sum(int(g.y.item()) for g in train_graphs),
            "val_pos": sum(int(g.y.item()) for g in val_graphs),
        }

        return train_loader, val_loader, stats
