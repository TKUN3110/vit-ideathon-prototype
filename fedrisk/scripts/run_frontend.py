"""
Launcher for FedRisk Streamlit Clinical Intelligence Dashboard.
Auto-detects project virtual environment (.venv) if available.
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Auto-detect and re-execute using project virtual environment (.venv)
venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe" if sys.platform == "win32" else BASE_DIR / ".venv" / "bin" / "python"
if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

DASHBOARD_PATH = BASE_DIR / "fedrisk" / "frontend" / "dashboard.py"


def main():
    print("=" * 70)
    print("[FedRisk] Launching Streamlit Clinical Dashboard on http://localhost:8501...")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_PATH),
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR))


if __name__ == "__main__":
    main()

