"""
Pydantic Request and Response Schemas for FedRisk API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrainingTriggerRequest(BaseModel):
    num_rounds: int = Field(default=5, ge=1, le=50, description="Total federated rounds")
    local_epochs: int = Field(default=2, ge=1, le=10, description="Epochs per client per round")
    learning_rate: float = Field(default=1e-3, gt=0, description="Local optimizer learning rate")
    use_smpc: bool = Field(default=True, description="Enable Secure Multi-Party Computation masking")


class TrainingStatusResponse(BaseModel):
    status: str = Field(description="Training status: idle, running, completed, error")
    current_round: int
    total_rounds: int
    elapsed_seconds: float
    latest_metrics: Dict[str, Any]
    smpc_active: bool


class SiteMetricItem(BaseModel):
    site_id: int
    site_name: str
    val_loss: float
    val_auroc: float
    val_auprc: float
    val_samples: int


class RoundMetricItem(BaseModel):
    round: int
    timestamp: str
    val_loss: float
    val_auroc: float
    val_auprc: float
    val_brier_score: float
    val_accuracy: float
    smpc_status: str
    site_metrics: List[SiteMetricItem] = []


class TrainingHistoryResponse(BaseModel):
    total_rounds_logged: int
    history: List[RoundMetricItem]


class PatientInferenceRequest(BaseModel):
    patient_id: Optional[str] = None
    site_id: Optional[int] = 0
    fhir_bundle: Optional[Dict[str, Any]] = None


class ClinicalEventItem(BaseModel):
    event_id: str
    code: str
    display: str
    relative_time_hours: float
    severity: str


class PatientRiskResponse(BaseModel):
    patient_id: str
    readmission_risk_score: float
    risk_percentage: float
    category: str
    clinical_recommendation: str
    event_count: int
    events: List[ClinicalEventItem] = []
    model_version: str


class HospitalNodeTelemetry(BaseModel):
    site_id: int
    name: str
    acuity_profile: str
    patient_count: int
    readmission_baseline: float
    latest_val_loss: Optional[float] = None
    latest_val_auroc: Optional[float] = None
    vram_fraction_allocated: float


class SystemHealthResponse(BaseModel):
    status: str
    torch_version: str
    cuda_available: bool
    gpu_name: str
    vram_total_mb: float
    vram_used_mb: float
    ram_total_gb: float
    ram_used_gb: float
    active_clients: int
