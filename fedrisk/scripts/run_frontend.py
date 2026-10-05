"""
Launcher for FedRisk High-Performance React Clinical Intelligence Platform.
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

FRONTEND_DIR = BASE_DIR / "fedrisk" / "frontend"


def main():
    print("=" * 70)
    print("[FedRisk] Launching Modern React Clinical Frontend on http://localhost:8501...")
    print("=" * 70)

    # Ensure node_modules are installed
    if not (FRONTEND_DIR / "node_modules").exists():
        print("[FedRisk] Installing frontend dependencies via npm...")
        subprocess.run(["npm.cmd" if sys.platform == "win32" else "npm", "install"], cwd=str(FRONTEND_DIR), check=True)

    cmd = ["npm.cmd" if sys.platform == "win32" else "npm", "run", "dev"]
    subprocess.run(cmd, cwd=str(FRONTEND_DIR))


if __name__ == "__main__":
    main()
