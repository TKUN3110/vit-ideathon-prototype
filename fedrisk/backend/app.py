"""
FedRisk FastAPI Server.
Provides endpoints for federated orchestration, real-time telemetry, and ICU readmission risk prediction.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List
import psutil
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ..config import (
    CHECKPOINT_DIR,
    CLIENT_GPU_FRACTION,
    HOSPITAL_METADATA,
    LOGS_DIR,
    NUM_FEDERATED_ROUNDS,
    PARTITIONS_DIR,
    SECURE_AGGREGATION_ENABLED,
)
from ..data.fhir_parser import FHIRParser
from ..data.graph_builder import PatientGraphBuilder
from ..data.partitioner import HospitalDataPartitioner
from ..federated.simulation import run_federated_simulation
from ..models.gnn import TemporalPatientRiskGNN
from .schemas import (
    ClinicalEventItem,
    HospitalNodeTelemetry,
    PatientInferenceRequest,
    PatientRiskResponse,
    RoundMetricItem,
    SystemHealthResponse,
    TrainingHistoryResponse,
    TrainingStatusResponse,
    TrainingTriggerRequest,
)
from .state import training_state

# Initialize FastAPI App
app = FastAPI(
    title="FedRisk ICU Readmission Platform",
    description="Privacy-Preserving ICU Readmission Risk Platform using PyG, Flower SMPC, and RTX 5060 Multiplexing.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Singletons
fhir_parser = FHIRParser()
graph_builder = PatientGraphBuilder()


def _async_training_worker(rounds: int, epochs: int, lr: float, use_smpc: bool):
    """Background worker thread executing Flower Simulation."""
    try:
        def on_round_complete(round_data: Dict[str, Any]):
            training_state.update_round(round_data)

        result = run_federated_simulation(
            num_rounds=rounds,
            state_callback=on_round_complete,
            use_gpu=True,
        )
        training_state.finish_training(result)
    except Exception as e:
        training_state.fail_training(str(e))
        print(f"[Backend Worker Error] {e}")


@app.get("/", tags=["General"])
def root():
    return {
        "platform": "FedRisk",
        "description": "Privacy-Preserving ICU Readmission Risk Prediction Platform",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.post("/api/orchestration/start", response_model=Dict[str, str], tags=["Federated Orchestration"])
def start_orchestration(req: TrainingTriggerRequest):
    """Triggers federated learning simulation across 3 hospital nodes."""
    success = training_state.start_training(total_rounds=req.num_rounds)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Federated training simulation is already actively running.",
        )

    worker = threading.Thread(
        target=_async_training_worker,
        args=(req.num_rounds, req.local_epochs, req.learning_rate, req.use_smpc),
        daemon=True,
    )
    training_state.worker_thread = worker
    worker.start()

    return {
        "message": f"Federated simulation started successfully for {req.num_rounds} rounds.",
        "status": "running",
    }


@app.post("/api/orchestration/restart", response_model=Dict[str, str], tags=["Federated Orchestration"])
def restart_orchestration(req: TrainingTriggerRequest):
    """Force-resets and restarts a fresh federated learning simulation."""
    training_state.reset_state()
    training_state.start_training(total_rounds=req.num_rounds, force=True)

    worker = threading.Thread(
        target=_async_training_worker,
        args=(req.num_rounds, req.local_epochs, req.learning_rate, req.use_smpc),
        daemon=True,
    )
    training_state.worker_thread = worker
    worker.start()

    return {
        "message": f"Federated simulation restarted successfully for {req.num_rounds} rounds.",
        "status": "running",
    }


@app.post("/api/orchestration/reset", response_model=Dict[str, Any], tags=["Federated Orchestration"])
def reset_orchestration():
    """Resets the active simulation state back to idle."""
    state = training_state.reset_state()
    return {
        "message": "Simulation state reset successfully to idle.",
        "status": "idle",
        "state": state,
    }


@app.get("/api/orchestration/status", response_model=TrainingStatusResponse, tags=["Federated Orchestration"])
def get_orchestration_status():
    """Returns real-time progress, loss, and AUROC of the active simulation."""
    return training_state.get_status()


@app.get("/api/orchestration/history", response_model=TrainingHistoryResponse, tags=["Federated Orchestration"])
def get_orchestration_history():
    """Retrieves full round-by-round convergence logs."""
    history = training_state.get_history()
    return {
        "total_rounds_logged": len(history),
        "history": history,
    }


@app.post("/api/predict/patient", response_model=PatientRiskResponse, tags=["Clinical Risk Inference"])
def predict_patient_risk(req: PatientInferenceRequest):
    """
    Evaluates ICU readmission risk for a clinical trajectory using the global GNN.
    Accepts either an explicit FHIR JSON Bundle or a patient ID from disk.
    """
    trajectory = None

    # Option A: Parse user-provided FHIR JSON Bundle
    if req.fhir_bundle:
        trajectory = fhir_parser.parse_bundle(req.fhir_bundle, hospital_site_id=req.site_id or 0)

    # Option B: Load from stored hospital partition
    elif req.patient_id:
        # Search partition raw FHIR directory
        site_id = req.site_id or 0
        site_raw_dir = PARTITIONS_DIR / f"site_{site_id}" / "raw_fhir"
        matched_file = None
        if site_raw_dir.exists():
            for f in site_raw_dir.glob("*.json"):
                if req.patient_id in f.stem or req.patient_id in f.name:
                    matched_file = f
                    break
        if matched_file:
            trajectory = fhir_parser.parse_bundle_file(matched_file, hospital_site_id=site_id)

    # Fallback: Generate demo patient trajectory
    if not trajectory:
        from ..data.dataset_generator import FHIRDatasetGenerator
        gen = FHIRDatasetGenerator()
        bundle = gen.generate_patient_bundle(999, req.site_id or 0, 0.5)
        trajectory = fhir_parser.parse_bundle(bundle, hospital_site_id=req.site_id or 0)

    # Construct PyG Graph
    patient_graph = graph_builder.trajectory_to_graph(trajectory)

    # Load Global Model Checkpoint
    model = TemporalPatientRiskGNN()
    best_ckpt = CHECKPOINT_DIR / "best_global_model.pt"
    latest_ckpt = CHECKPOINT_DIR / "latest_global_model.pt"

    model_version = "Untrained Baseline"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, weights_only=False))
        model_version = "Best Global Checkpoint (AUROC Optimized)"
    elif latest_ckpt.exists():
        model.load_state_dict(torch.load(latest_ckpt, weights_only=False))
        model_version = "Latest Federated Round Checkpoint"

    model.eval()
    risk_info = model.predict_risk(patient_graph)

    # Format clinical events
    event_items = [
        ClinicalEventItem(
            event_id=ev.event_id,
            code=ev.code,
            display=ev.display,
            relative_time_hours=ev.relative_time_hours,
            severity=ev.severity,
        )
        for ev in trajectory.events
    ]

    return PatientRiskResponse(
        patient_id=trajectory.patient_id,
        readmission_risk_score=risk_info["readmission_risk_score"],
        risk_percentage=risk_info["risk_percentage"],
        category=risk_info["category"],
        clinical_recommendation=risk_info["clinical_recommendation"],
        event_count=len(event_items),
        events=event_items,
        model_version=model_version,
    )


@app.get("/api/nodes/telemetry", response_model=List[HospitalNodeTelemetry], tags=["Telemetry & Telematics"])
def get_nodes_telemetry():
    """Returns telemetry and clinical profiles across the 3 hospital partitions."""
    latest_metrics = training_state.latest_metrics.get("site_metrics", [])
    metric_map = {m["site_id"]: m for m in latest_metrics if "site_id" in m}

    nodes = []
    for site_id, meta in HOSPITAL_METADATA.items():
        site_m = metric_map.get(site_id, {})
        nodes.append(
            HospitalNodeTelemetry(
                site_id=site_id,
                name=meta["name"],
                acuity_profile=meta["acuity_profile"],
                patient_count=meta["patient_count"],
                readmission_baseline=meta["readmission_baseline"],
                latest_val_loss=site_m.get("val_loss"),
                latest_val_auroc=site_m.get("val_auroc"),
                resource_fraction_allocated=CLIENT_GPU_FRACTION,
            )
        )
    return nodes


@app.get("/api/system/health", response_model=SystemHealthResponse, tags=["Telemetry & Telematics"])
def get_system_health():
    """System health monitor."""
    cuda_avail = torch.cuda.is_available()
    mem = psutil.virtual_memory()
    ram_total_gb = round(mem.total / (1024**3), 2)
    ram_used_gb = round(mem.used / (1024**3), 2)

    return SystemHealthResponse(
        status="operational",
        cuda_available=cuda_avail,
        ram_total_gb=ram_total_gb,
        ram_used_gb=ram_used_gb,
        active_clients=3,
    )



@app.get("/api/patients/list", tags=["Clinical Risk Inference"])
def list_patients(site_id: int = 0):
    """Lists available patient records for a specific hospital partition."""
    site_raw_dir = PARTITIONS_DIR / f"site_{site_id}" / "raw_fhir"
    if not site_raw_dir.exists():
        # Generate mock dataset if missing
        from ..data.dataset_generator import FHIRDatasetGenerator
        FHIRDatasetGenerator().generate_partitioned_dataset()

    patient_files = sorted(list(site_raw_dir.glob("*.json"))) if site_raw_dir.exists() else []
    results = []
    
    for p_file in patient_files:
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            traj = fhir_parser.parse_bundle(bundle, hospital_site_id=site_id)
            primary_diag = traj.events[0].display if traj.events else "Unspecified ICU Admission"
            results.append({
                "patient_id": traj.patient_id,
                "gender": traj.gender,
                "length_of_stay_hours": traj.length_of_stay_hours,
                "event_count": len(traj.events),
                "readmitted_30d": traj.readmitted_30d,
                "primary_diagnosis": primary_diag,
                "site_id": site_id,
                "file_name": p_file.name
            })
        except Exception as e:
            continue

    return {
        "site_id": site_id,
        "site_name": HOSPITAL_METADATA.get(site_id, {}).get("name", f"Site-{site_id}"),
        "count": len(results),
        "patients": results
    }


@app.get("/api/patients/graph/{patient_id}", tags=["Clinical Risk Inference"])
def get_patient_graph_topology(patient_id: str, site_id: int = 0):
    """Retrieves full patient clinical event DAG topology and risk prediction."""
    site_raw_dir = PARTITIONS_DIR / f"site_{site_id}" / "raw_fhir"
    matched_file = None

    patient_num = patient_id.split("-")[-1] if "-" in patient_id else patient_id

    if site_raw_dir.exists():
        for f in site_raw_dir.glob("*.json"):
            if patient_id in f.stem or patient_id in f.name or f.stem.endswith(patient_num) or f"patient_{patient_num}" in f.stem:
                matched_file = f
                break

    if not matched_file:
        # Fallback search across all site directories
        for s in range(3):
            s_dir = PARTITIONS_DIR / f"site_{s}" / "raw_fhir"
            if s_dir.exists():
                for f in s_dir.glob("*.json"):
                    if patient_id in f.stem or patient_id in f.name or f.stem.endswith(patient_num) or f"patient_{patient_num}" in f.stem:
                        matched_file = f
                        site_id = s
                        break
            if matched_file:
                break

    if not matched_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient ID {patient_id} not found in hospital partitions.",
        )

    with open(matched_file, "r", encoding="utf-8") as f:
        bundle_data = json.load(f)

    trajectory = fhir_parser.parse_bundle(bundle_data, hospital_site_id=site_id)
    patient_graph = graph_builder.trajectory_to_graph(trajectory)

    # Load Global GNN Checkpoint
    model = TemporalPatientRiskGNN()
    best_ckpt = CHECKPOINT_DIR / "best_global_model.pt"
    latest_ckpt = CHECKPOINT_DIR / "latest_global_model.pt"

    model_version = "Untrained Baseline"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, weights_only=False))
        model_version = "Best Global Checkpoint (AUROC Optimized)"
    elif latest_ckpt.exists():
        model.load_state_dict(torch.load(latest_ckpt, weights_only=False))
        model_version = "Latest Federated Round Checkpoint"

    model.eval()
    risk_info = model.predict_risk(patient_graph)

    # Format DAG nodes & edges
    nodes = []
    for i, ev in enumerate(trajectory.events):
        nodes.append({
            "id": i,
            "event_id": ev.event_id,
            "code": ev.code,
            "label": ev.display,
            "relative_time_hours": ev.relative_time_hours,
            "severity": ev.severity,
            "encounter_id": ev.encounter_id,
        })

    edges = []
    edge_index = patient_graph.edge_index.cpu().numpy()
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        if u != v:  # Exclude self loops from network view
            edges.append({"source": u, "target": v})

    return {
        "patient_id": trajectory.patient_id,
        "site_id": site_id,
        "site_name": HOSPITAL_METADATA.get(site_id, {}).get("name", f"Site-{site_id}"),
        "gender": trajectory.gender,
        "length_of_stay_hours": trajectory.length_of_stay_hours,
        "readmitted_30d": trajectory.readmitted_30d,
        "readmission_risk_score": risk_info["readmission_risk_score"],
        "risk_percentage": risk_info["risk_percentage"],
        "category": risk_info["category"],
        "clinical_recommendation": risk_info["clinical_recommendation"],
        "model_version": model_version,
        "nodes": nodes,
        "edges": edges,
        "events": [
            {
                "event_id": ev.event_id,
                "code": ev.code,
                "display": ev.display,
                "relative_time_hours": ev.relative_time_hours,
                "severity": ev.severity,
                "encounter_id": ev.encounter_id,
            }
            for ev in trajectory.events
        ]
    }

