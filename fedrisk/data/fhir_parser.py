"""
FHIR Parser for Intensive Care Unit (ICU) Patient Trajectories.
Parses FHIR R4 JSON schemas (Patient, Condition, Encounter) into structured clinical event sequences.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ClinicalEvent:
    event_id: str
    code: str
    display: str
    category: str
    onset_datetime: datetime
    relative_time_hours: float
    severity: str
    encounter_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedPatientTrajectory:
    patient_id: str
    gender: str
    birth_date: Optional[str]
    admission_time: datetime
    discharge_time: Optional[datetime]
    length_of_stay_hours: float
    readmitted_30d: int
    events: List[ClinicalEvent]
    hospital_site_id: int = 0


class FHIRParser:
    """
    Standard FHIR R4 Ingestion Engine.
    Converts FHIR Bundles and discrete JSON resources (Patient, Condition, Encounter)
    into normalized temporal patient trajectories.
    """

    def __init__(self):
        self.date_formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]

    def _parse_datetime(self, dt_str: str) -> datetime:
        if not dt_str:
            return datetime.utcnow()
        for fmt in self.date_formats:
            try:
                # Truncate timezone offset if parsing without timezone
                clean_str = dt_str.replace("Z", "+00:00")
                if fmt.endswith("%z") and "+" in clean_str:
                    return datetime.strptime(clean_str, fmt)
                return datetime.strptime(dt_str.split("+")[0].replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
            except (ValueError, IndexError):
                continue
        # Fallback to current timestamp if parsing fails
        return datetime.utcnow()

    def parse_bundle(self, bundle_dict: Dict[str, Any], hospital_site_id: int = 0) -> Optional[ParsedPatientTrajectory]:
        """
        Parses a single FHIR Bundle containing Patient, Encounter, and Condition resources.
        """
        patient_resource = None
        encounters: List[Dict[str, Any]] = []
        conditions: List[Dict[str, Any]] = []

        entries = bundle_dict.get("entry", [])
        for entry in entries:
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType")
            if rtype == "Patient":
                patient_resource = resource
            elif rtype == "Encounter":
                encounters.append(resource)
            elif rtype == "Condition":
                conditions.append(resource)

        if not patient_resource:
            return None

        patient_id = patient_resource.get("id", "unknown_patient")
        gender = patient_resource.get("gender", "unknown")
        birth_date = patient_resource.get("birthDate")

        # Determine index ICU encounter
        if not encounters:
            # Fallback default encounter if none provided
            index_encounter = {
                "id": "enc_default",
                "period": {
                    "start": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                    "end": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                },
                "readmission": 0,
            }
        else:
            # Sort encounters chronologically
            encounters.sort(key=lambda e: self._parse_datetime(e.get("period", {}).get("start", "")))
            index_encounter = encounters[0]

        start_time_str = index_encounter.get("period", {}).get("start", "")
        end_time_str = index_encounter.get("period", {}).get("end", "")

        admission_time = self._parse_datetime(start_time_str)
        discharge_time = self._parse_datetime(end_time_str) if end_time_str else None

        los_hours = (
            (discharge_time - admission_time).total_seconds() / 3600.0
            if discharge_time
            else 24.0
        )
        if los_hours < 1.0:
            los_hours = 24.0

        # Readmission flag check (either in encounter metadata or extension)
        readmission_flag = 0
        if "readmission_30d" in index_encounter:
            readmission_flag = int(index_encounter["readmission_30d"])
        elif "hospitalization" in index_encounter:
            hosp = index_encounter["hospitalization"]
            if hosp.get("reAdmission", {}).get("coding", [{}])[0].get("code") == "R":
                readmission_flag = 1
        elif len(encounters) > 1:
            # Multiple encounters within 30 days of discharge indicate readmission
            second_encounter = encounters[1]
            sec_start = self._parse_datetime(second_encounter.get("period", {}).get("start", ""))
            if discharge_time and 0 < (sec_start - discharge_time).total_seconds() <= (30 * 86400):
                readmission_flag = 1

        # Parse condition events
        events: List[ClinicalEvent] = []
        for cond in conditions:
            cond_id = cond.get("id", f"cond_{len(events)}")
            coding_list = cond.get("code", {}).get("coding", [{}])
            primary_coding = coding_list[0] if coding_list else {}
            code = primary_coding.get("code", "R69")  # R69 = Illness, unspecified
            display = primary_coding.get("display", cond.get("code", {}).get("text", "Medical Event"))
            
            category = "diagnosis"
            cat_list = cond.get("category", [])
            if cat_list and "coding" in cat_list[0]:
                category = cat_list[0]["coding"][0].get("code", "diagnosis")

            onset_str = cond.get("onsetDateTime") or start_time_str
            onset_dt = self._parse_datetime(onset_str)

            # Compute relative time from ICU admission in hours
            rel_hours = (onset_dt - admission_time).total_seconds() / 3600.0
            if rel_hours < 0:
                rel_hours = 0.0

            severity = cond.get("severity", {}).get("coding", [{}])[0].get("code", "moderate")
            enc_ref = cond.get("encounter", {}).get("reference", index_encounter.get("id", ""))

            event = ClinicalEvent(
                event_id=cond_id,
                code=code,
                display=display,
                category=category,
                onset_datetime=onset_dt,
                relative_time_hours=rel_hours,
                severity=severity,
                encounter_id=enc_ref,
                attributes={"status": cond.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "active")},
            )
            events.append(event)

        # Sort events chronologically by relative onset time
        events.sort(key=lambda ev: ev.relative_time_hours)

        return ParsedPatientTrajectory(
            patient_id=patient_id,
            gender=gender,
            birth_date=birth_date,
            admission_time=admission_time,
            discharge_time=discharge_time,
            length_of_stay_hours=los_hours,
            readmitted_30d=readmission_flag,
            events=events,
            hospital_site_id=hospital_site_id,
        )

    def parse_bundle_file(self, file_path: Union[str, Path], hospital_site_id: int = 0) -> Optional[ParsedPatientTrajectory]:
        """Loads and parses a FHIR JSON file from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"FHIR bundle file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.parse_bundle(data, hospital_site_id=hospital_site_id)
