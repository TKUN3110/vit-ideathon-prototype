"""
Thread-Safe Global Telemetry and Training State Manager for FedRisk.
"""

import threading
import time
from typing import Any, Dict, List, Optional


class GlobalTrainingState:
    """
    Singleton repository managing asynchronous federated simulation status,
    historical metrics, and clinical inference checkpoints.
    """

    _instance: Optional["GlobalTrainingState"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.status = "idle"  # "idle" | "running" | "completed" | "error"
        self.current_round = 0
        self.total_rounds = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.latest_metrics: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.smpc_audits: List[Dict[str, Any]] = []
        self.error_message: Optional[str] = None
        self.worker_thread: Optional[threading.Thread] = None

    def start_training(self, total_rounds: int, force: bool = False) -> bool:
        with self._lock:
            if self.status == "running" and not force:
                return False
            self.status = "running"
            self.current_round = 0
            self.total_rounds = total_rounds
            self.start_time = time.time()
            self.end_time = None
            self.latest_metrics = {}
            self.history = []
            self.error_message = None
            return True

    def reset_state(self) -> Dict[str, Any]:
        """Resets the simulation status back to idle, allowing fresh runs."""
        with self._lock:
            self.status = "idle"
            self.current_round = 0
            self.total_rounds = 0
            self.start_time = None
            self.end_time = None
            self.latest_metrics = {}
            self.history = []
            self.error_message = None
            return self._get_status_unlocked()

    def update_round(self, round_data: Dict[str, Any]) -> None:
        with self._lock:
            self.current_round = round_data.get("round", self.current_round)
            self.latest_metrics = round_data
            self.history.append(round_data)
            # When all requested rounds have been reported, immediately mark as completed
            # so the UI never appears stuck while background cleanup finishes
            if self.total_rounds > 0 and self.current_round >= self.total_rounds:
                self.status = "completed"
                self.end_time = time.time()

    def finish_training(self, result_meta: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self.status = "completed"
            if not self.end_time:
                self.end_time = time.time()
            if result_meta and "smpc_audits" in result_meta:
                self.smpc_audits = result_meta["smpc_audits"]

    def fail_training(self, err: str) -> None:
        with self._lock:
            self.status = "error"
            self.end_time = time.time()
            self.error_message = str(err)

    def _get_status_unlocked(self) -> Dict[str, Any]:
        # Self-heal: If total rounds have completed but status is still marked running, complete it
        if self.total_rounds > 0 and self.current_round >= self.total_rounds and self.status == "running":
            self.status = "completed"
            if not self.end_time:
                self.end_time = time.time()

        elapsed = 0.0
        if self.start_time:
            end = self.end_time if self.end_time else time.time()
            elapsed = round(end - self.start_time, 1)

        return {
            "status": self.status,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "elapsed_seconds": elapsed,
            "latest_metrics": self.latest_metrics,
            "error_message": self.error_message,
            "smpc_active": True,
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return self._get_status_unlocked()

    def get_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.history)


# Global instance
training_state = GlobalTrainingState()

