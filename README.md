# FedRisk: Privacy-Preserving ICU Readmission Prediction Platform

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-flwr-FF9800.svg)](https://flower.ai/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch%20Geometric-3C2179.svg)](https://pyg.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)

FedRisk is an enterprise-grade, privacy-preserving ICU readmission prediction platform engineered for high-acuity intensive care environments. It leverages **PyTorch Geometric (PyG)** to model patient trajectories as dynamic temporal event graphs, **Flower (flwr)** to execute federated training across 3 simulated hospital nodes with GPU hardware acceleration, **SMPC (Secure Multi-Party Computation / Secure Aggregation)** to prevent clinical weight leakage, a **FastAPI** backend for lifecycle orchestration, and a modern **Streamlit** clinical intelligence dashboard.

---

## Environment & Architecture

- **Cross-Platform Compute**: Supports CUDA GPU acceleration and standard CPU execution.
- **True Multi-Node Isolation**: Employs a multi-container Docker architecture where each hospital node runs in total isolation. They communicate with the central aggregator securely over the network.
- **Native Synthea Generation**: Employs the official Synthea Java engine to procedurally generate ABDM-compliant, highly realistic longitudinal EHR data.

---

## Repository Structure

```
.
├── fedrisk/
│   ├── config.py                   # Central configuration module
│   ├── data/
│   │   ├── fhir_parser.py          # FHIR R4 JSON parser (Patient, Condition, Encounter) + PyHealth
│   │   ├── graph_builder.py        # Converts EHR trajectories into PyG temporal event DAGs
│   │   └── dataset_generator.py    # Generates realistic synthetic ICU cohorts via Synthea Java Engine
│   ├── models/
│   │   ├── gnn.py                  # TemporalPatientRiskGNN (GATv2 + Edge Features + Dual Readout)
│   │   └── metrics.py              # AUROC, AUPRC, Brier Score, and calibration metrics
│   ├── federated/
│   │   ├── client.py               # flwr.client.NumPyClient with AMP and memory optimizations
│   │   └── strategy.py             # SMPC / Secure Aggregation strategy with zero-sum masking
│   ├── backend/
│   │   └── app.py                  # FastAPI server for orchestration & risk inference
│   ├── frontend/
│   │   └── dashboard.py            # Streamlit interactive UI dashboard
│   └── scripts/
│       ├── run_server.py           # Starts the central Flower aggregator
│       ├── run_client.py           # Starts a local hospital node for federated training
│       ├── run_backend.py          # FastAPI launcher
│       └── run_frontend.py         # Streamlit launcher
├── requirements.txt                # Full Python dependencies
├── Dockerfile.backend              # Docker build config for backend and clients
├── Dockerfile.frontend             # Docker build config for the Streamlit dashboard
├── docker-compose.yml              # Multi-node simulation architecture configuration
└── README.md                       # Platform documentation
```

---

## Clinical Data Pipeline & Dynamic Graph Topology

The platform ingests standard **FHIR R4 JSON** bundles representing longitudinal intensive care admissions:
1. **`Patient`**: Demographics, age, biological sex.
2. **`Condition`**: Discrete clinical diagnoses mapped to ICD-10 codes (e.g., Sepsis `A41.9`, ARDS `J80`), onset timestamps, severity, and clinical status. Uses **PyHealth** for standard ontology mapping.
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

### 1. Docker Container Deployment (Recommended)
Build and run the entire multi-node architecture (Aggregator, 3 Clients, FastAPI, Streamlit) in isolated Docker containers:
```bash
docker compose up --build
```
- **Streamlit Clinical Dashboard**: http://localhost:8501
- **Backend API & Orchestrator**: http://localhost:8000

*(Note: Data is mounted to `./data`. Checkpoints are mounted to `./checkpoints`.)*

### 2. Run Locally (Without Docker)

**A. Generate Synthetic FHIR Datasets (Requires Java)**
Generate 300 synthetic patient records partitioned across 3 hospital sites using Synthea:
```bash
python fedrisk/data/dataset_generator.py
```

**B. Launch FastAPI Backend**
```bash
python fedrisk/scripts/run_backend.py
```

**C. Launch Streamlit Clinical Dashboard**
In a separate terminal:
```bash
python fedrisk/scripts/run_frontend.py
```

---

## REST API Specification

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/orchestration/start` | Triggers federated simulation asynchronously across nodes |
| `GET` | `/api/orchestration/status` | Returns active round, elapsed time, loss, and AUROC |
| `GET` | `/api/orchestration/history` | Retrieves full round-by-round convergence time-series |
| `POST` | `/api/predict/patient` | Evaluates readmission risk for a clinical trajectory or FHIR bundle |
| `GET` | `/api/patients/list` | Returns list of patient trajectories for a hospital partition |
| `GET` | `/api/patients/graph/{patient_id}` | Retrieves full patient DAG topology and risk score |
| `GET` | `/api/nodes/telemetry` | Returns telemetry for Site-A, Site-B, and Site-C |
| `GET` | `/api/system/health` | System compute and RAM health monitor |
