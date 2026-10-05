"""
Synthea FHIR Dataset Generator for Federated ICU Readmission.
Uses Synthea to generate real-world mapped clinical health records.
Loads them into SQLite, parses, and partitions them into 3 cohorts.
"""

import json
import sqlite3
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List
import torch
import glob

from ..config import HOSPITAL_METADATA, PARTITIONS_DIR
from .fhir_parser import FHIRParser
from .graph_builder import PatientGraphBuilder

class FHIRDatasetGenerator:
    def __init__(self, seed: int = 42):
        import random
        self.seed = seed
        random.seed(seed)
        self.parser = FHIRParser()
        self.graph_builder = PatientGraphBuilder()
        self.db_path = Path("data/hospital_records.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS fhir_bundles (id TEXT PRIMARY KEY, site_id INTEGER, bundle_json TEXT)")
        
    def generate_partitioned_dataset(self) -> Dict[int, List[Dict[str, Any]]]:
        print("[FHIR Generator] Generating real-world mapped clinical health data using Synthea...")
        # Clean previous output
        if Path("output").exists():
            shutil.rmtree("output")
            
        # Run Synthea (generate around 30 patients for speed, adjust as needed)
        # Assuming synthea.jar is at the root
        try:
            subprocess.run(["java", "-jar", "synthea.jar", "-p", "30"], check=True)
        except Exception as e:
            print(f"[FHIR Generator] Error running Synthea: {e}")
            print("[FHIR Generator] Ensure synthea.jar is in the root directory and Java is installed.")
            return {}

        fhir_files = glob.glob("output/fhir/*.json")
        # Filter out practitioner/hospital info files, keep only patient bundles
        patient_files = [f for f in fhir_files if "practitioner" not in f.lower() and "hospital" not in f.lower()]
        
        all_partitions: Dict[int, List[Dict[str, Any]]] = {}
        
        # Distribute patients among 3 sites
        for site_id, meta in HOSPITAL_METADATA.items():
            site_dir = PARTITIONS_DIR / f"site_{site_id}"
            raw_dir = site_dir / "raw_fhir"
            site_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            all_partitions[site_id] = []

        site_idx = 0
        for p_file in patient_files:
            with open(p_file, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            
            patient_id = bundle.get("entry", [{}])[0].get("resource", {}).get("id", Path(p_file).stem)
            
            # Store in SQLite
            self.conn.execute(
                "INSERT OR REPLACE INTO fhir_bundles (id, site_id, bundle_json) VALUES (?, ?, ?)",
                (patient_id, site_idx, json.dumps(bundle))
            )
            self.conn.commit()
            
            # Write to partition dir
            site_dir = PARTITIONS_DIR / f"site_{site_idx}"
            raw_dir = site_dir / "raw_fhir"
            bundle_path = raw_dir / f"{patient_id}.json"
            with open(bundle_path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, indent=2)
                
            all_partitions[site_idx].append(bundle)
            site_idx = (site_idx + 1) % len(HOSPITAL_METADATA)
            
        # Build graphs
        for site_id, meta in HOSPITAL_METADATA.items():
            site_dir = PARTITIONS_DIR / f"site_{site_id}"
            site_graphs = []
            
            for bundle in all_partitions[site_id]:
                trajectory = self.parser.parse_bundle(bundle, hospital_site_id=site_id)
                if trajectory:
                    graph_data = self.graph_builder.trajectory_to_graph(trajectory)
                    site_graphs.append(graph_data)
                    
            if site_graphs:
                graphs_path = site_dir / "graphs.pt"
                torch.save(site_graphs, graphs_path)
                
                readmitted_sum = sum(g.y.item() for g in site_graphs)
                readmit_rate = (readmitted_sum / len(site_graphs)) * 100.0 if site_graphs else 0.0
                print(f"[FHIR Generator] Processed {len(site_graphs)} Synthea graphs for {meta['name']} -> Readmission Rate: {readmit_rate:.1f}%")

        return all_partitions
