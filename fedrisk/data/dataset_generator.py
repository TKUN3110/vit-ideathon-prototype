"""
Synthetic FHIR Dataset Generator for Federated ICU Readmission.
Generates compliant FHIR R4 Bundles (Patient, Condition, Encounter) and partitions them
into 3 realistic non-IID hospital cohorts (Trauma/Tertiary, Cardiovascular, Community).
"""

import glob
import json
import random
import shutil
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from ..config import HOSPITAL_METADATA, PARTITIONS_DIR
from .fhir_parser import FHIRParser
from .graph_builder import CLINICAL_CODE_VOCAB, PatientGraphBuilder


SITE_CLINICAL_PROFILES = {
    0: {  # Site-A: Metro Trauma & Tertiary Center
        "primary_conditions": [
            ("S06.9X0A", "Traumatic brain injury", "severe", 0.9),
            ("R65.21", "Severe sepsis with septic shock", "severe", 0.95),
            ("J80", "Acute respiratory distress syndrome (ARDS)", "severe", 0.92),
            ("N17.9", "Acute kidney failure, unspecified", "moderate", 0.8),
            ("D65", "Disseminated intravascular coagulation (DIC)", "severe", 0.88),
            ("Z99.11", "Dependence on respirator [ventilator]", "severe", 0.85),
        ],
        "secondary_conditions": [
            ("R57.2", "Septic shock", "severe"),
            ("I95.9", "Hypotension, unspecified", "moderate"),
            ("R09.02", "Hypoxemia", "moderate"),
            ("T81.4XXA", "Infection following a procedure", "moderate"),
        ],
    },
    1: {  # Site-B: Heart & Vascular Institute
        "primary_conditions": [
            ("I21.9", "Acute myocardial infarction", "severe", 0.85),
            ("R57.0", "Cardiogenic shock", "severe", 0.92),
            ("I50.9", "Heart failure, unspecified", "moderate", 0.75),
            ("I48.91", "Atrial fibrillation", "moderate", 0.65),
            ("I63.9", "Cerebral infarction, unspecified", "severe", 0.84),
        ],
        "secondary_conditions": [
            ("I95.9", "Hypotension, unspecified", "moderate"),
            ("R00.0", "Tachycardia, unspecified", "mild"),
            ("N17.9", "Acute kidney failure, unspecified", "moderate"),
            ("R09.02", "Hypoxemia", "moderate"),
        ],
    },
    2: {  # Site-C: Community Memorial Hospital
        "primary_conditions": [
            ("A41.9", "Sepsis, unspecified organism", "moderate", 0.70),
            ("J96.00", "Acute respiratory failure", "moderate", 0.75),
            ("E11.10", "Type 2 diabetes with ketoacidosis", "moderate", 0.72),
            ("R69", "Illness, unspecified", "mild", 0.35),
            ("I50.9", "Heart failure, unspecified", "moderate", 0.65),
        ],
        "secondary_conditions": [
            ("R00.0", "Tachycardia, unspecified", "mild"),
            ("I95.9", "Hypotension, unspecified", "moderate"),
            ("R09.02", "Hypoxemia", "mild"),
            ("R57.9", "Shock, unspecified", "moderate"),
        ],
    },
}


class FHIRDatasetGenerator:
    """
    Simulates high-fidelity electronic health records compliant with FHIR R4 schemas.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.parser = FHIRParser()
        self.graph_builder = PatientGraphBuilder()
        self.db_path = Path("data/hospital_records.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS fhir_bundles (id TEXT PRIMARY KEY, site_id INTEGER, bundle_json TEXT)"
        )

    def generate_patient_bundle(
        self,
        patient_idx: int,
        site_id: int,
        readmission_prob: float,
    ) -> Dict[str, Any]:
        """
        Creates a valid FHIR R4 Bundle containing Patient, Encounter, and Condition resources.
        """
        patient_id = f"PAT-SITE{site_id}-{patient_idx:04d}"
        encounter_id = f"ENC-SITE{site_id}-{patient_idx:04d}"

        # Temporal anchor: ICU admission occurred between 10 to 60 days ago
        base_time = datetime.utcnow() - timedelta(days=random.randint(10, 60), hours=random.randint(0, 23))
        los_hours = round(random.uniform(24.0, 168.0), 1)  # 1 to 7 days stay
        discharge_time = base_time + timedelta(hours=los_hours)

        # Correlate readmission outcome with acuity profile
        is_readmitted = 1 if random.random() < readmission_prob else 0

        # Patient Demographics Resource
        gender = random.choice(["male", "female"])
        birth_year = random.randint(1945, 1995)
        birth_date = f"{birth_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

        patient_resource = {
            "resourceType": "Patient",
            "id": patient_id,
            "active": True,
            "gender": gender,
            "birthDate": birth_date,
            "managingOrganization": {"reference": f"Organization/Hospital-Site-{site_id}"},
        }

        # Encounter Resource (ICU Inpatient Admission)
        encounter_resource = {
            "resourceType": "Encounter",
            "id": encounter_id,
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "IMP",
                "display": "inpatient encounter",
            },
            "serviceType": {
                "coding": [{"code": "ICU", "display": "Intensive Care Unit"}]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "period": {
                "start": base_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": discharge_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "hospitalization": {
                "reAdmission": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0092",
                            "code": "R" if is_readmitted else "N",
                            "display": "Readmission within 30 days" if is_readmitted else "No readmission",
                        }
                    ]
                }
            },
            "readmission_30d": is_readmitted,
        }

        # Condition Resources (Temporal clinical trajectory during ICU stay)
        conditions: List[Dict[str, Any]] = []
        site_profile = SITE_CLINICAL_PROFILES.get(site_id, SITE_CLINICAL_PROFILES[0])

        # Select primary condition
        primary_pool = site_profile["primary_conditions"]
        prim_code, prim_display, prim_sev, _ = random.choice(primary_pool)

        # Primary condition diagnosed upon admission (t = 0)
        cond_0 = {
            "resourceType": "Condition",
            "id": f"COND-{patient_id}-00",
            "clinicalStatus": {"coding": [{"code": "active", "display": "Active"}]},
            "verificationStatus": {"coding": [{"code": "confirmed", "display": "Confirmed"}]},
            "category": [
                {"coding": [{"code": "encounter-diagnosis", "display": "Encounter Diagnosis"}]}
            ],
            "severity": {"coding": [{"code": prim_sev, "display": prim_sev.capitalize()}]},
            "code": {
                "coding": [
                    {
                        "system": "http://hl7.org/fhir/sid/icd-10-cm",
                        "code": prim_code,
                        "display": prim_display,
                    }
                ],
                "text": prim_display,
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "encounter": {"reference": f"Encounter/{encounter_id}"},
            "onsetDateTime": base_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        conditions.append(cond_0)

        # Number of secondary complications scales with readmission status
        num_secondary = random.randint(2, 5) if is_readmitted else random.randint(1, 3)
        secondary_pool = site_profile["secondary_conditions"]

        for c_idx in range(1, num_secondary + 1):
            sec_choice = random.choice(secondary_pool)
            sec_code, sec_display, sec_sev = sec_choice

            # Complications emerge progressively over the ICU course
            onset_delta_hours = random.uniform(2.0, min(los_hours, 72.0))
            cond_time = base_time + timedelta(hours=onset_delta_hours)

            cond_res = {
                "resourceType": "Condition",
                "id": f"COND-{patient_id}-{c_idx:02d}",
                "clinicalStatus": {"coding": [{"code": "active", "display": "Active"}]},
                "verificationStatus": {"coding": [{"code": "confirmed", "display": "Confirmed"}]},
                "category": [
                    {"coding": [{"code": "problem-list-item", "display": "Problem List"}]}
                ],
                "severity": {"coding": [{"code": sec_sev, "display": sec_sev.capitalize()}]},
                "code": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10-cm",
                            "code": sec_code,
                            "display": sec_display,
                        }
                    ],
                    "text": sec_display,
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "encounter": {"reference": f"Encounter/{encounter_id}"},
                "onsetDateTime": cond_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            conditions.append(cond_res)

        # Assemble into standard FHIR R4 Bundle
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "id": f"BUNDLE-{patient_id}",
            "meta": {"lastUpdated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
            "entry": [
                {"fullUrl": f"urn:uuid:patient-{patient_id}", "resource": patient_resource},
                {"fullUrl": f"urn:uuid:encounter-{encounter_id}", "resource": encounter_resource},
            ]
            + [{"fullUrl": f"urn:uuid:condition-{c['id']}", "resource": c} for c in conditions],
        }

        return bundle

    def generate_partitioned_dataset(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Generates and persists synthetic cohorts for Site 0, 1, and 2.
        Saves FHIR JSON bundles, populates SQLite, and builds PyG graphs to disk.
        """
        all_partitions: Dict[int, List[Dict[str, Any]]] = {}

        for site_id, meta in HOSPITAL_METADATA.items():
            site_dir = PARTITIONS_DIR / f"site_{site_id}"
            raw_dir = site_dir / "raw_fhir"
            site_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)

            patient_count = meta["patient_count"]
            baseline_risk = meta["readmission_baseline"]

            site_bundles: List[Dict[str, Any]] = []
            site_graphs = []

            for i in range(patient_count):
                bundle = self.generate_patient_bundle(i, site_id, baseline_risk)
                site_bundles.append(bundle)

                patient_id = bundle["entry"][0]["resource"]["id"]

                # Store in SQLite
                self.conn.execute(
                    "INSERT OR REPLACE INTO fhir_bundles (id, site_id, bundle_json) VALUES (?, ?, ?)",
                    (patient_id, site_id, json.dumps(bundle)),
                )

                # Save individual FHIR JSON
                bundle_path = raw_dir / f"patient_{i:04d}.json"
                with open(bundle_path, "w", encoding="utf-8") as f:
                    json.dump(bundle, f, indent=2)

                # Parse and build graph
                trajectory = self.parser.parse_bundle(bundle, hospital_site_id=site_id)
                if trajectory:
                    graph_data = self.graph_builder.trajectory_to_graph(trajectory)
                    site_graphs.append(graph_data)

            self.conn.commit()

            # Persist serialized PyG graph dataset
            graphs_path = site_dir / "graphs.pt"
            torch.save(site_graphs, graphs_path)

            readmitted_sum = sum(g.y.item() for g in site_graphs)
            readmit_rate = (readmitted_sum / len(site_graphs)) * 100.0 if site_graphs else 0.0

            print(
                f"[FHIR Generator] Generated {len(site_graphs)} graphs for {meta['name']} -> "
                f"Readmission Rate: {readmit_rate:.1f}% | Saved to {site_dir}"
            )
            all_partitions[site_id] = site_bundles

        return all_partitions
