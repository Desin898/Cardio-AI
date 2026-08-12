import sqlite3
from pathlib import Path
from typing import Generator
from backend.app.core.config import settings

DB_PATH = settings.PROJECT_ROOT / "coronary_ai.db"

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()
