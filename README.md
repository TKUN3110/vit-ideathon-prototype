# FedRisk: Privacy-Preserving ICU Readmission Prediction Platform

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-flwr[simulation]-FF9800.svg)](https://flower.ai/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch%20Geometric-3C2179.svg)](https://pyg.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg?logo=react)](https://react.dev/)

FedRisk is an enterprise-grade, privacy-preserving ICU readmission prediction platform engineered for high-acuity intensive care environments. It leverages **PyTorch Geometric (PyG)** to model patient trajectories as dynamic temporal event graphs, **Flower (flwr[simulation])** to execute federated training across 3 simulated hospital nodes with GPU hardware acceleration, **SMPC (Secure Multi-Party Computation / Secure Aggregation)** to prevent clinical weight leakage, a **FastAPI** backend for lifecycle orchestration, and a modern **React** clinical intelligence dashboard.

---

## Environment & Architecture

- **Cross-Platform Compute**: Supports CUDA GPU acceleration and standard CPU execution.
- **Resource Multiplexing**: Flower's Simulation Engine partitions compute resources across simulated hospital nodes with client resource limits (`client_resources={"num_cpus": 1, "num_gpus": 0.33}`). Local training loops invoke Automatic Mixed Precision (`torch.amp.autocast('cuda')`) to optimize memory efficiency.
- **Containerized & Native Deployment Options**: Supports containerized deployment via **Docker & Docker Compose** for isolated node execution as well as native local environment execution.

---

## Repository Structure

```
.
├── fedrisk/
│   ├── __init__.py                 # FedRisk package root
│   ├── config.py                   # Central configuration module
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
│   │   ├── client.py               # flwr.client.NumPyClient with AMP and memory optimizations
│   │   ├── strategy.py             # SMPC / Secure Aggregation strategy with zero-sum masking
│   │   └── simulation.py           # Flower Simulation Engine multiplexing client nodes
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI server for orchestration & risk inference
│   │   ├── schemas.py              # Pydantic data schemas
│   │   └── state.py                # Thread-safe global training telemetry repository
│   ├── frontend/
│   │   ├── package.json            # React Vite dependencies
│   │   ├── vite.config.js          # Vite configuration
│   │   └── src/                    # React UI components, Patient DAG Canvas & Risk Gauge
│   └── scripts/
│       ├── __init__.py
│       ├── generate_mock_data.py   # Script to generate 3-site FHIR datasets
│       ├── run_simulation.py       # Standalone CLI federated simulation executor
│       ├── run_backend.py          # FastAPI launcher
│       └── run_frontend.py         # React clinical dashboard launcher
├── requirements.txt                # Full Python dependencies
└── README.md                       # Platform documentation
```

---

## Clinical Data Pipeline & Dynamic Graph Topology

The platform ingests standard **FHIR R4 JSON** bundles representing longitudinal intensive care admissions:
1. **`Patient`**: Demographics, age, biological sex.
2. **`Condition`**: Discrete clinical diagnoses mapped to ICD-10 codes (e.g., Sepsis `A41.9`, ARDS `J80`, Septic Shock `R57.2`, AMI `I21.9`), onset timestamps, severity, and clinical status.
3. **`Encounter`**: ICU admission intervals, length of stay, and 30-day readmission indicators.

---

## Hospital Cohort Partitions (Non-IID Distribution)

FedRisk partitions data across 3 heterogeneous clinical sites:
- **Site 0 (Site-A: Metro Trauma & Tertiary Center)**: High-complexity trauma, severe sepsis, ARDS, mechanical ventilation. Higher baseline readmission rate (~35%).
- **Site 1 (Site-B: Heart & Vascular Institute)**: Acute myocardial infarction, cardiogenic shock, heart failure, atrial fibrillation. Baseline readmission (~25%).
- **Site 2 (Site-C: Community Memorial Hospital)**: Diabetic ketoacidosis, COPD, general respiratory failure, pneumonia. Baseline readmission (~18%).

---

## Secure Multi-Party Computation (SMPC / SecAgg)

To protect patient privacy and comply with HIPAA/GDPR, local gradient vectors are blinded using **Zero-Sum Additive Secret Sharing**:
$$\tilde{w}_i = w_i + \sum_{j > i} R_{ij} - \sum_{j < i} R_{ji}$$
When all hospital nodes submit their encrypted updates to the central aggregator:
$$\sum_{i=1}^3 \tilde{w}_i = \sum_{i=1}^3 w_i \quad \text{since} \quad \sum_{i=1}^3 \text{mask}_i = 0$$
- The central aggregator **never sees individual hospital gradients**, mitigating model inversion attacks.
- Residual mask norm $||\sum \text{mask}_i||_2 < 10^{-6}$ is audited after every federated round.

---

## Quickstart Guide

### 1. Environment Verification
```bash
python -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```

### 2. Generate Synthetic FHIR Datasets
Generate 300 synthetic patient records partitioned across 3 hospital sites:
```bash
python fedrisk/scripts/generate_mock_data.py
```

### 3. Run Federated Simulation (CLI)
Execute a 5-round federated training run:
```bash
python fedrisk/scripts/run_simulation.py --rounds 5
```

### 4. Launch FastAPI Backend
```bash
python fedrisk/scripts/run_backend.py
# Interactive API Docs: http://127.0.0.1:8000/docs
```

### 5. Launch React Clinical Dashboard
In a separate terminal:
```bash
python fedrisk/scripts/run_frontend.py
# Web Dashboard: http://localhost:8501
```

### 6. Docker Container Deployment (Multi-Node Containerization)
Build and run all services in isolated Docker containers:
```bash
docker compose up --build
```
- **Backend API & Orchestrator**: http://localhost:8000
- **React Dashboard**: http://localhost:8501

---

## REST API Specification

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/orchestration/start` | Triggers federated simulation asynchronously across 3 nodes |
| `GET` | `/api/orchestration/status` | Returns active round, elapsed time, loss, and AUROC |
| `GET` | `/api/orchestration/history` | Retrieves full round-by-round convergence time-series |
| `POST` | `/api/predict/patient` | Evaluates readmission risk for a clinical trajectory or FHIR bundle |
| `GET` | `/api/patients/list` | Returns list of patient trajectories for a hospital partition |
| `GET` | `/api/patients/graph/{patient_id}` | Retrieves full patient DAG topology and risk score |
| `GET` | `/api/nodes/telemetry` | Returns telemetry for Site-A, Site-B, and Site-C |
| `GET` | `/api/system/health` | System compute and RAM health monitor |
