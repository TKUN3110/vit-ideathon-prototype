"""
Launcher for FedRisk Streamlit Clinical Dashboard.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DASHBOARD_PATH = BASE_DIR / "fedrisk" / "frontend" / "dashboard.py"


def main():
    print("=" * 70)
    print("[FedRisk] Launching Streamlit Dashboard on port 8501...")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_PATH),
        "--server.port=8501",
        "--server.headless=true",
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
