"""
Patient Graph Builder.
Transforms discrete FHIR patient trajectories into PyTorch Geometric (PyG) Data graphs
capturing dynamic topological, temporal, and clinical semantic relationships.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data

from ..config import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from .fhir_parser import ClinicalEvent, ParsedPatientTrajectory

# High-acuity ICU ICD-10 Code Vocabulary with Clinical Severity Prior
CLINICAL_CODE_VOCAB: Dict[str, Tuple[int, float]] = {
    # Code: (Vocab Index, Clinical Acuity Weight [0.0 - 1.0])
    "A41.9": (1, 0.90),   # Sepsis, unspecified organism
    "R57.0": (2, 0.95),   # Cardiogenic shock
    "R57.2": (3, 0.98),   # Septic shock
    "R57.9": (4, 0.85),   # Shock, unspecified
    "J80":   (5, 0.92),   # Acute respiratory distress syndrome (ARDS)
    "J96.00":(6, 0.88),   # Acute respiratory failure
    "I50.9": (7, 0.75),   # Heart failure, unspecified
    "I21.9": (8, 0.85),   # Acute myocardial infarction
    "I48.91":(9, 0.65),   # Atrial fibrillation
    "N17.9": (10, 0.82),  # Acute kidney failure, unspecified
    "K72.00":(11, 0.86),  # Acute hepatic failure
    "E11.10":(12, 0.78),  # Type 2 diabetes with ketoacidosis
    "I63.9": (13, 0.84),  # Cerebral infarction, unspecified
    "T81.4XXA":(14, 0.70),# Infection following a procedure
    "I95.9": (15, 0.60),  # Hypotension, unspecified
    "R09.02":(16, 0.72),  # Hypoxemia
    "R00.0": (17, 0.50),  # Tachycardia, unspecified
    "R65.21":(18, 0.96),  # Severe sepsis with septic shock
    "D65":   (19, 0.89),  # Disseminated intravascular coagulation (DIC)
    "S06.9X0A":(20, 0.85),# Traumatic brain injury
    "Z99.11":(21, 0.80),  # Dependence on respirator [ventilator]
    "R69":   (22, 0.30),  # Illness, unspecified
}

SEVERITY_WEIGHT_MAP = {
    "high": 1.0,
    "severe": 1.0,
    "moderate": 0.6,
    "mild": 0.3,
    "low": 0.2,
    "active": 0.5,
}


class PatientGraphBuilder:
    """
    Constructs PyTorch Geometric Data graphs from clinical trajectories.
    Nodes = Clinical events / diagnoses.
    Edges = Directed temporal transitions + contemporaneous co-occurrence relations.
    """

    def __init__(self, node_dim: int = NODE_FEATURE_DIM, edge_dim: int = EDGE_FEATURE_DIM):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.vocab_size = len(CLINICAL_CODE_VOCAB) + 10

        # Deterministic projection seed for code embeddings
        torch.manual_seed(42)
        self.code_embedding_table = torch.nn.Embedding(self.vocab_size + 1, self.node_dim - 4)

    def _encode_node_features(self, event: ClinicalEvent) -> torch.Tensor:
        """
        Generates a feature vector for a discrete clinical event node:
        [Code_Embedding (dim-4), Normalized_Time (1), Acuity_Weight (1), Severity_Weight (1), Log_LOS (1)]
        """
        code_entry = CLINICAL_CODE_VOCAB.get(event.code)
        if code_entry:
            vocab_idx, acuity_weight = code_entry
        else:
            # Hash unseen code into vocab space safely
            vocab_idx = (abs(hash(event.code)) % (self.vocab_size - 1)) + 1
            acuity_weight = 0.40

        with torch.no_grad():
            code_emb = self.code_embedding_table(torch.tensor(vocab_idx, dtype=torch.long))

        # Normalized time relative to 7 days (168 hours) ICU horizon
        norm_time = float(np.clip(event.relative_time_hours / 168.0, 0.0, 1.0))
        severity_val = SEVERITY_WEIGHT_MAP.get(str(event.severity).lower(), 0.5)

        # Log-transformed time to capture rapid onset in first 24h
        log_time = float(np.log1p(event.relative_time_hours) / np.log1p(168.0))

        aux_features = torch.tensor(
            [norm_time, acuity_weight, severity_val, log_time],
            dtype=torch.float32,
        )

        return torch.cat([code_emb, aux_features], dim=0)

    def trajectory_to_graph(self, trajectory: ParsedPatientTrajectory) -> Data:
        """
        Converts a parsed trajectory into a PyG Data instance.
        """
        events = trajectory.events

        # Fallback if patient has no recorded events: create initial admission node
        if not events:
            events = [
                ClinicalEvent(
                    event_id="admit_fallback",
                    code="R69",
                    display="ICU Baseline Admission",
                    category="observation",
                    onset_datetime=trajectory.admission_time,
                    relative_time_hours=0.0,
                    severity="moderate",
                    encounter_id="enc_init",
                )
            ]

        num_nodes = len(events)
        node_features = [self._encode_node_features(ev) for ev in events]
        x = torch.stack(node_features, dim=0)  # Shape: [num_nodes, node_dim]

        # Construct Edge Topology:
        # 1. Temporal Directed Edges: from earlier events to subsequent events
        # 2. Intra-encounter / Contemporaneous co-occurrence edges (within 4 hours)
        # 3. Self-loops for message passing stability
        edge_indices: List[List[int]] = []
        edge_attributes: List[List[float]] = []

        for i in range(num_nodes):
            # Self-loop
            edge_indices.append([i, i])
            edge_attributes.append([0.0, 0.0, 1.0, 0.0])  # [dt, type=self, decay=1.0, direction=0]

            for j in range(num_nodes):
                if i == j:
                    continue

                dt_hours = events[j].relative_time_hours - events[i].relative_time_hours
                dt_norm = float(np.clip(abs(dt_hours) / 168.0, 0.0, 1.0))
                time_decay = float(math.exp(-abs(dt_hours) / 24.0))

                # Temporal Forward Edge: event i precedes event j
                if dt_hours > 0.0:
                    # Connect immediate successors or close temporal context (< 48 hrs)
                    if (j == i + 1) or (dt_hours <= 48.0):
                        edge_indices.append([i, j])
                        edge_attributes.append([dt_norm, 1.0, time_decay, 1.0])

                # Contemporaneous / Co-occurrence: within 3 hours
                elif abs(dt_hours) <= 3.0:
                    edge_indices.append([i, j])
                    edge_attributes.append([dt_norm, 2.0, time_decay, 0.0])

        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attributes, dtype=torch.float32)

        # Label: ICU 30-Day Readmission (1 or 0)
        y_tensor = torch.tensor([float(trajectory.readmitted_30d)], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            y=y_tensor,
            patient_id=trajectory.patient_id,
            num_nodes=num_nodes,
        )

        # Metadata for clinical explainability in frontend
        data.event_names = [ev.display for ev in events]
        data.event_codes = [ev.code for ev in events]
        data.onset_hours = [ev.relative_time_hours for ev in events]
        data.site_id = trajectory.hospital_site_id

        return data
