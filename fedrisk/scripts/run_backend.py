"""
Launcher for FedRisk FastAPI Backend Server.
Auto-detects and uses project virtual environment (.venv) if available.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Auto-detect and re-execute using project virtual environment (.venv)
venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe" if sys.platform == "win32" else BASE_DIR / ".venv" / "bin" / "python"
if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from fedrisk.config import BACKEND_HOST, BACKEND_PORT


def main():
    print("=" * 70)
    print(f"[FedRisk] Starting FastAPI Backend on http://{BACKEND_HOST}:{BACKEND_PORT}")
    print(f"Interactive API Docs available at: http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
    print("=" * 70)

    from fedrisk.backend.app import app

    uvicorn.run(
        app,
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
