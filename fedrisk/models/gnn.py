"""
Dynamic Patient Graph Neural Network (TemporalPatientRiskGNN).
Processes dynamic clinical event topologies to output patient risk embeddings and ICU readmission probabilities.
Optimized for mixed-precision (FP16) execution.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from ..config import (
    DROPOUT_RATE,
    EDGE_FEATURE_DIM,
    GNN_HEADS,
    HIDDEN_DIM,
    NODE_FEATURE_DIM,
    NUM_CLASSES,
)


class TemporalPatientRiskGNN(nn.Module):
    """
    Graph Attention Network tailored for temporal ICU patient event DAGs.
    Captures temporal progression, condition co-occurrence, and acute shock onset.
    """

    def __init__(
        self,
        node_in_dim: int = NODE_FEATURE_DIM,
        edge_in_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        heads: int = GNN_HEADS,
        dropout: float = DROPOUT_RATE,
        out_dim: int = NUM_CLASSES,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout

        # Initial node and edge projections
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Message Passing Layer 1 (Multi-Head GATv2 with Edge Features)
        head_dim = hidden_dim // heads
        self.conv1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=heads,
            edge_dim=hidden_dim,
            concat=True,
            dropout=dropout,
            add_self_loops=False,  # GraphBuilder already builds self-loops
        )
        self.ln1 = nn.LayerNorm(hidden_dim)

        # Message Passing Layer 2
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=heads,
            edge_dim=hidden_dim,
            concat=True,
            dropout=dropout,
            add_self_loops=False,
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Readout: Combines Mean (Overall baseline acuity) and Max (Peak acute event)
        self.readout_dim = hidden_dim * 2

        # Clinical Risk Classification Head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(self.readout_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Node features [N, node_in_dim]
            edge_index: Graph topology [2, E]
            edge_attr: Temporal edge features [E, edge_in_dim]
            batch: PyG batch indicator [N]
        Returns:
            logits: Unnormalized risk predictions [B, 1]
            risk_embeddings: Patient representation vectors [B, readout_dim]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Encode input features
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        # Layer 1 with Residual Connection
        h_res1 = h
        h = self.conv1(h, edge_index, edge_attr=e)
        h = self.ln1(h + h_res1)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2 with Residual Connection
        h_res2 = h
        h = self.conv2(h, edge_index, edge_attr=e)
        h = self.ln2(h + h_res2)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Global Readout / Graph Pooling
        pool_mean = global_mean_pool(h, batch)
        pool_max = global_max_pool(h, batch)
        patient_embedding = torch.cat([pool_mean, pool_max], dim=-1)  # [B, hidden_dim * 2]

        # Classification Logits
        logits = self.classifier(patient_embedding)  # [B, 1]

        return logits, patient_embedding

    def predict_risk(self, data: Data) -> Dict[str, float]:
        """
        Inference helper for an individual patient graph.
        Returns probability score and qualitative clinical triage level.
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

            logits, _ = self.forward(x, edge_index, edge_attr, batch)
            prob = torch.sigmoid(logits).item()

        # Clinical Triage Classification
        if prob < 0.25:
            category = "Low Risk"
            recommendation = "Standard ICU step-down to medical/surgical floor. Routine monitoring."
        elif prob < 0.50:
            category = "Moderate Risk"
            recommendation = "Extended telemetry observation. Post-discharge nurse follow-up within 48h."
        elif prob < 0.75:
            category = "High Risk"
            recommendation = "Delay discharge. Re-evaluate respiratory index and renal biomarkers."
        else:
            category = "Critical Risk"
            recommendation = "Intensive monitoring retained. High probability of acute relapse within 30 days."

        return {
            "readmission_risk_score": round(prob, 4),
            "risk_percentage": round(prob * 100.0, 2),
            "category": category,
            "clinical_recommendation": recommendation,
        }
