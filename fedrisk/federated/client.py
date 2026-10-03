"""
Federated Hospital Node Client (flwr.client.NumPyClient).
Executes local GNN training on partitioned FHIR patient graphs with RTX 5060 memory management.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from ..config import (
    CLIENT_GPU_FRACTION,
    HOSPITAL_METADATA,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    SECURE_AGGREGATION_ENABLED,
    TARGET_GPU_DEVICE,
    USE_AMP,
    USE_CUDA,
    WEIGHT_DECAY,
)
from ..data.partitioner import HospitalDataPartitioner
from ..models.gnn import TemporalPatientRiskGNN
from ..models.metrics import ClinicalMetricsTracker


class FedRiskClient(fl.client.NumPyClient):
    """
    Simulated hospital node executing local patient graph learning.
    Multiplexes the RTX 5060 GPU and safely clears VRAM cache upon completion.
    """

    def __init__(self, site_id: int):
        self.site_id = site_id
        self.site_name = HOSPITAL_METADATA.get(site_id, {}).get("name", f"Site-{site_id}")

        # Device determination: allocate to RTX 5060 if available
        if USE_CUDA and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{TARGET_GPU_DEVICE}")
        else:
            self.device = torch.device("cpu")

        # Instantiate local model
        self.model = TemporalPatientRiskGNN().to(self.device)

        # Load partitioned hospital graphs
        self.partitioner = HospitalDataPartitioner()
        self.train_loader, self.val_loader, self.stats = self.partitioner.get_site_dataloaders(self.site_id)

        # Dynamic positive class weighting to combat ICU readmission class imbalance
        pos_ratio = (self.stats["train_pos"] / self.stats["train_samples"]) if self.stats["train_samples"] > 0 else 0.25
        weight_val = (1.0 - pos_ratio) / max(pos_ratio, 1e-3)
        self.pos_weight = torch.tensor([min(weight_val, 4.0)], dtype=torch.float32, device=self.device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Extracts model weights as NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Loads updated global weights into the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, device=self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """
        Executes local GNN optimization on hospital patient graphs.
        """
        self.set_parameters(parameters)
        self.model.train()

        epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        lr = float(config.get("lr", LEARNING_RATE))
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        scaler = torch.amp.GradScaler('cuda') if (USE_AMP and self.device.type == "cuda") else None

        train_tracker = ClinicalMetricsTracker()

        for epoch in range(epochs):
            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits, _ = self.model(
                            batch.x,
                            batch.edge_index,
                            batch.edge_attr,
                            batch.batch,
                        )
                        loss = self.loss_fn(logits.view(-1), batch.y.view(-1))
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, _ = self.model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch,
                    )
                    loss = self.loss_fn(logits.view(-1), batch.y.view(-1))
                    loss.backward()
                    optimizer.step()

                # Track batch metrics
                with torch.no_grad():
                    probs = torch.sigmoid(logits).view(-1).cpu().tolist()
                    targets = batch.y.view(-1).cpu().tolist()
                    train_tracker.update(targets, probs, loss.item(), batch.num_graphs)

        fit_metrics = train_tracker.compute()
        fit_metrics["site_id"] = self.site_id

        # Extract updated parameters
        updated_params = self.get_parameters(config={})

        # Memory Cleanup: Flush RTX 5060 VRAM after training epoch cycle
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return updated_params, self.stats["train_samples"], fit_metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluates current global weights against local validation patient graphs.
        """
        self.set_parameters(parameters)
        self.model.eval()

        val_tracker = ClinicalMetricsTracker()

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                if USE_AMP and self.device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        logits, _ = self.model(
                            batch.x,
                            batch.edge_index,
                            batch.edge_attr,
                            batch.batch,
                        )
                        loss = self.loss_fn(logits.view(-1), batch.y.view(-1))
                else:
                    logits, _ = self.model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch,
                    )
                    loss = self.loss_fn(logits.view(-1), batch.y.view(-1))

                probs = torch.sigmoid(logits).view(-1).cpu().tolist()
                targets = batch.y.view(-1).cpu().tolist()
                val_tracker.update(targets, probs, loss.item(), batch.num_graphs)

        metrics = val_tracker.compute()
        metrics["site_id"] = self.site_id

        # Memory Cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return float(metrics["loss"]), self.stats["val_samples"], metrics
