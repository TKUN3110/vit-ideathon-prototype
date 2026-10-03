"""
FedRisk - Central Configuration Module
Hardware-aware settings tailored for NVIDIA RTX 5060 (8GB VRAM) and 16GB Host RAM.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
FEDRISK_DIR = BASE_DIR / "fedrisk"
DATA_DIR = BASE_DIR / "data"
PARTITIONS_DIR = DATA_DIR / "partitions"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
for directory in [DATA_DIR, PARTITIONS_DIR, CHECKPOINT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hardware & Device Settings
# Optimized for RTX 5060 (8GB VRAM) and 16GB System RAM
# Prevents GPU/RAM OOM by allocating 0.33 GPU per client in Flower Simulation
USE_CUDA = True  # Will fallback gracefully if CUDA is not available
TARGET_GPU_DEVICE = 0
NUM_CLIENTS = 3
CLIENT_GPU_FRACTION = 0.33  # Multiplex single RTX 5060 across 3 simulated hospital nodes
CLIENT_CPU_COUNT = 1
USE_AMP = True  # Automatic Mixed Precision (FP16) to conserve VRAM

# GNN Architecture Hyperparameters
NODE_FEATURE_DIM = 32
HIDDEN_DIM = 64
EDGE_FEATURE_DIM = 4
GNN_HEADS = 4
GNN_LAYERS = 2
DROPOUT_RATE = 0.2
NUM_CLASSES = 1  # Binary prediction: ICU 30-Day Readmission

# Federated Learning Settings
NUM_FEDERATED_ROUNDS = 5
LOCAL_EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Privacy & Security (SMPC / Secure Aggregation)
SECURE_AGGREGATION_ENABLED = True
SMPC_MODULUS = 2**31 - 1  # Mersenne prime for modular integer arithmetic if discrete SMPC used
SMPC_MASK_SCALE = 1e5     # Fixed-point quantization factor

# Hospital Partitions Meta
HOSPITAL_METADATA = {
    0: {
        "name": "Site-A (Metro Trauma & Tertiary Center)",
        "acuity_profile": "High Acuity (Trauma, Multi-Organ Failure)",
        "patient_count": 120,
        "readmission_baseline": 0.35,
    },
    1: {
        "name": "Site-B (Heart & Vascular Institute)",
        "acuity_profile": "Cardiovascular & Post-Operative ICU",
        "patient_count": 100,
        "readmission_baseline": 0.25,
    },
    2: {
        "name": "Site-C (Community Memorial Hospital)",
        "acuity_profile": "Geriatric & General Medical ICU",
        "patient_count": 80,
        "readmission_baseline": 0.18,
    },
}

# Network & Server Endpoints
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_PORT = 8501
