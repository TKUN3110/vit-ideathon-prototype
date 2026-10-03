"""
Launcher for FedRisk FastAPI Backend Server.
"""

import sys
from pathlib import Path
import uvicorn

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fedrisk.config import BACKEND_HOST, BACKEND_PORT


def main():
    print("=" * 70)
    print(f"[FedRisk] Starting FastAPI Backend on http://{BACKEND_HOST}:{BACKEND_PORT}")
    print(f"Interactive API Docs available at: http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
    print("=" * 70)

    uvicorn.run(
        "fedrisk.backend.app:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
