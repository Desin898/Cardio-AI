import os
import sys
import time
import socket
import threading
import subprocess
import logging
from typing import Dict, Any

from backend.app.core.config import settings
from backend.app.engines.base_engine import BaseMLEngine

class DeepSAEngine(BaseMLEngine):
    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()

    def load_models(self) -> None:
        pass

    def is_running(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", settings.DEEPSA_PORT), timeout=1):
                return True
        except OSError:
            return False

    def ensure_running(self) -> None:
        with self._lock:
            if self.is_running():
                return
            logging.info(f"Starting DeepSA process from: {settings.DEEPSA_SCRIPT}")
            self._proc = subprocess.Popen(
                [sys.executable, str(settings.DEEPSA_SCRIPT)],
                cwd=str(settings.PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.is_running():
                return
            time.sleep(0.5)
        raise RuntimeError(
            "DeepSA (demo.py) failed to start within 30 seconds. "
            "Check that the model checkpoint exists and all dependencies are installed."
        )

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_running()
        return {"status": "running", "url": settings.DEEPSA_URL}

deepsa_engine = DeepSAEngine()
