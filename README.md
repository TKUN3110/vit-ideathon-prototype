# FedRisk: Privacy-Preserving ICU Readmission Prediction Platform

[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA->=12.8-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![Flower](https://img.shields.io/badge/Flower-flwr[simulation]-FF9800.svg)](https://flower.ai/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch%20Geometric-3C2179.svg)](https://pyg.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)

FedRisk is an enterprise-grade, privacy-preserving ICU readmission prediction platform engineered for high-acuity intensive care environments. It leverages **PyTorch Geometric (PyG)** to model patient trajectories as dynamic temporal event graphs, **Flower (flwr[simulation])** to execute federated training across 3 simulated hospital nodes multiplexing an NVIDIA GeForce RTX 5060 GPU, **SMPC (Secure Multi-Party Computation / Secure Aggregation)** to prevent clinical weight leakage, a **FastAPI** backend for lifecycle orchestration, and an interactive **Streamlit** clinical intelligence dashboard.

---

## Hardware & Environment Architecture

- **Host Hardware**: 16GB System RAM, NVIDIA GeForce RTX 5060 Laptop/Desktop GPU (~8GB VRAM).
- **PyTorch & CUDA**: `torch>=2.4.0` (with CUDA >= 12.8 build: `torch-2.11.0+cu128`).
- **Resource Multiplexing**: Flower's Simulation Engine partitions the RTX 5060 across 3 simulated hospital nodes with `client_resources={"num_cpus": 1, "num_gpus": 0.33}`. Local training loops invoke `torch.cuda.empty_cache()` and Automatic Mixed Precision (`torch.amp.autocast('cuda')`) to prevent host RAM exhaustion and GPU OOM.
- **Docker-Free Infrastructure**: Runs directly on native Python 3.10+ environments without requiring containerization.

---

## Repository Structure

```
.
├── fedrisk/
│   ├── __init__.py                 # FedRisk package root
│   ├── config.py                   # Hardware thresholds (RTX 5060), GNN hyperparams, paths
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fhir_parser.py          # FHIR R4 JSON parser (Patient, Condition, Encounter)
│   │   ├── graph_builder.py        # Converts EHR trajectories into PyG temporal event DAGs
│   │   ├── dataset_generator.py    # Generates realistic synthetic ICU cohorts across 3 sites
│   │   └── partitioner.py          # PyG DataLoader factory & non-IID train/val splitter
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn.py                  # TemporalPatientRiskGNN (GATv2 + Edge Features + Dual Readout)
│   │   └── metrics.py              # AUROC, AUPRC, Brier Score, and calibration metrics
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── client.py               # flwr.client.NumPyClient with VRAM emptying & AMP
│   │   ├── strategy.py             # SMPC / Secure Aggregation strategy with zero-sum masking
│   │   └── simulation.py           # Flower Simulation Engine multiplexing RTX 5060 (0.33 GPU/node)
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI server for orchestration & risk inference
│   │   ├── schemas.py              # Pydantic data schemas
│   │   └── state.py                # Thread-safe global training telemetry repository
│   ├── frontend/
│   │   ├── __init__.py
│   │   └── dashboard.py            # Streamlit clinical dashboard & patient graph explorer
│   └── scripts/
│       ├── __init__.py
│       ├── generate_mock_data.py   # Script to generate 3-site FHIR datasets
│       ├── run_simulation.py       # Standalone CLI federated simulation executor
│       ├── run_backend.py          # FastAPI launcher
│       └── run_frontend.py         # Streamlit dashboard launcher
├── requirements.txt                # Full dependencies
└── README.md                       # Platform documentation
```

---

## Clinical Data Pipeline & Dynamic Graph Topology

The platform ingests standard **FHIR R4 JSON** bundles representing longitudinal intensive care admissions:
1. **`Patient`**: Demographics, age, biological sex.
2. **`Condition`**: Discrete clinical diagnoses mapped to ICD-10 codes (e.g., Sepsis `A41.9`, ARDS `J80`, Septic Shock `R57.2`, AMI `I21.9`), onset timestamps, severity, and clinical status.
3. **`Encounter`**: ICU admission intervals, length of stay, and 30-day readmission indicators.

### Graph Representation in PyG:
- **Nodes**: Individual clinical events / diagnoses. Node features $\mathbf{x} \in \mathbb{R}^{N \times 32}$ integrate semantic code embeddings, normalized time relative to ICU admission ($t/168\text{h}$), clinical acuity weights, and log-transformed onset latency.
- **Edges**: Directed temporal transitions $(u \to v)$ where condition $u$ precedes condition $v$, plus contemporaneous co-occurrence edges (diagnoses within the same 3-hour clinical window). Edge features $\mathbf{e} \in \mathbb{R}^{E \times 4}$ encode $\Delta t$, connection type, and temporal decay $\exp(-\Delta t / 24)$.
- **Target ($y$)**: Binary indicator for 30-day ICU readmission.

---

## Hospital Cohort Partitions (Non-IID Distribution)

FedRisk partitions data across 3 realistic, heterogeneous clinical sites:
- **Site 0 (Site-A: Metro Trauma & Tertiary Center)**: High-complexity trauma, severe sepsis, ARDS, mechanical ventilation. Higher baseline readmission rate (~35%).
- **Site 1 (Site-B: Heart & Vascular Institute)**: Acute myocardial infarction, cardiogenic shock, heart failure, atrial fibrillation. Baseline readmission (~25%).
- **Site 2 (Site-C: Community Memorial Hospital)**: Diabetic ketoacidosis, COPD, general respiratory failure, pneumonia. Baseline readmission (~18%).

---

## Secure Multi-Party Computation (SMPC / SecAgg)

To protect patient privacy and comply with HIPAA/GDPR, local gradient vectors are blinded using **Zero-Sum Additive Secret Sharing**:
$$\tilde{w}_i = w_i + \sum_{j > i} R_{ij} - \sum_{j < i} R_{ji}$$
When all hospital nodes submit their encrypted updates to the central aggregator:
$$\sum_{i=1}^3 \tilde{w}_i = \sum_{i=1}^3 w_i \quad \text{since} \quad \sum_{i=1}^3 \text{mask}_i = 0$$
- The central aggregator **never sees individual hospital gradients**, completely mitigating model inversion and membership inference attacks.
- Residual mask norm $||\sum \text{mask}_i||_2 < 10^{-6}$ is audited after every federated round.

---

## Quickstart Guide

### 1. Environment Setup
```bash
# Clone or navigate to the repository
cd c:\Users\tejas\Desktop\Coding\nigger

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Verify PyTorch CUDA 12.8 + RTX 5060
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Generate Synthetic FHIR Datasets
Generate 300 synthetic patient records partitioned across the 3 hospital sites:
```bash
python fedrisk/scripts/generate_mock_data.py
```

### 3. Run Federated Simulation (CLI)
Execute a 5-round federated training run multiplexing the RTX 5060:
```bash
python fedrisk/scripts/run_simulation.py --rounds 5
```

### 4. Launch FastAPI Backend
```bash
python fedrisk/scripts/run_backend.py
# Docs: http://127.0.0.1:8000/docs
```

### 5. Launch Streamlit Clinical Dashboard
In a separate terminal:
```bash
python fedrisk/scripts/run_frontend.py
# Dashboard: http://localhost:8501
```

---

## REST API Specification

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/orchestration/start` | Triggers federated simulation asynchronously across 3 nodes |
| `GET` | `/api/orchestration/status` | Returns active round, elapsed time, loss, and AUROC |
| `GET` | `/api/orchestration/history` | Retrieves full round-by-round convergence time-series |
| `POST` | `/api/predict/patient` | Evaluates readmission risk for a clinical trajectory or FHIR bundle |
| `GET` | `/api/nodes/telemetry` | Returns telemetry for Site-A, Site-B, and Site-C |
| `GET` | `/api/system/health` | Hardware health monitor (RTX 5060 VRAM and 16GB RAM) |
