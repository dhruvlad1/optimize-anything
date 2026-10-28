"""
trackH_dbms/db.py
-----------------
Database connection configuration for Track H (DBMS Optimization).

Reads connection parameters from environment variables:
  PGHOST      - default: localhost
  PGPORT      - default: 5432
  PGDATABASE  - default: optdb
  PGUSER      - default: optuser
  PGPASSWORD  - REQUIRED; raises RuntimeError if absent (never written to disk)
"""

import os
import psycopg  # psycopg3


def _get_config() -> dict:
    """Return a dict of connection parameters read from environment variables."""
    password = os.environ.get("PGPASSWORD")
    if not password:
        raise RuntimeError(
            "PGPASSWORD environment variable is not set. "
            "Set it before running any Track H script, e.g.:\n"
            "  $env:PGPASSWORD = 'your_password'  # PowerShell\n"
            "  export PGPASSWORD=your_password     # bash"
        )
    return {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": int(os.environ.get("PGPORT", "5432")),
        "dbname": os.environ.get("PGDATABASE", "optdb"),
        "user": os.environ.get("PGUSER", "optuser"),
        "password": password,
    }


def connect() -> psycopg.Connection:
    """Open and return a new psycopg3 Connection using environment variables.

    Raises
    ------
    RuntimeError
        If PGPASSWORD is not set.
    psycopg.OperationalError
        If the connection cannot be established.
    """
    cfg = _get_config()
    return psycopg.connect(**cfg)
