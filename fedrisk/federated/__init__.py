"""
FedRisk Federated Learning Package.
Features Flower Simulation Engine, SMPC Secure Aggregation, and Clinical Telemetry.
"""

from .client import FedRiskClient
from .strategy import FedRiskSecAggStrategy
from .simulation import run_federated_simulation

__all__ = [
    "FedRiskClient",
    "FedRiskSecAggStrategy",
    "run_federated_simulation",
]
