"""
trackH_dbms/tests/test_db.py
-----------------------------
Test suite for Track H (DBMS Optimization).

All tests are automatically skipped when:
  - PGPASSWORD is not set, or
  - the database is not reachable.

Run:
    pytest trackH_dbms/tests -v
"""

from __future__ import annotations

import hashlib
import os
import pytest

# ---------------------------------------------------------------------------
# Skip guard — must come before any DB import
# ---------------------------------------------------------------------------

def _db_available() -> tuple[bool, str]:
    if not os.environ.get("PGPASSWORD"):
        return False, "PGPASSWORD is not set"
    try:
        import psycopg
        from trackH_dbms.db import connect
        conn = connect()
        conn.close()
        return True, ""
    except Exception as exc:
        return False, str(exc)


_DB_OK, _DB_REASON = _db_available()
skip_if_no_db = pytest.mark.skipif(not _DB_OK, reason=f"DB unavailable: {_DB_REASON}")


# ---------------------------------------------------------------------------
# Expected row counts — spec numbers (scale factor 0.1), written directly here
# ---------------------------------------------------------------------------

EXPECTED_COUNTS = {
    "region":   5,
    "nation":  25,
    "supplier": 1_000,
    "customer": 15_000,
    "part":     20_000,
    "orders":   150_000,
}
# lineitem is variable (Poisson 1–7 per order, avg ~4): check a range.
LINEITEM_MIN = 570_000
LINEITEM_MAX = 630_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md5_rows(rows: list) -> str:
    h = hashlib.md5()
    for row in sorted(str(r) for r in rows):
        h.update(row.encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@skip_if_no_db
def test_row_counts():
    """All tables have the expected number of rows after data load."""
    from trackH_dbms.db import connect
    conn = connect()
    with conn.cursor() as cur:
        for table, expected in EXPECTED_COUNTS.items():
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            actual = cur.fetchone()[0]
            assert actual == expected, (
                f"{table}: expected {expected} rows, got {actual}"
            )

        cur.execute("SELECT COUNT(*) FROM lineitem")
        li_count = cur.fetchone()[0]
        assert LINEITEM_MIN <= li_count <= LINEITEM_MAX, (
            f"lineitem row count {li_count} outside expected range "
            f"[{LINEITEM_MIN}, {LINEITEM_MAX}]"
        )
    conn.close()


@skip_if_no_db
def test_only_pk_indexes():
    """No secondary indexes exist — only primary-key constraints."""
    from trackH_dbms.db import connect
    conn = connect()
    tables = list(EXPECTED_COUNTS.keys()) + ["lineitem"]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tablename, indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = ANY(%s)
              AND indexname NOT LIKE '%%_pkey'
            """,
            (tables,),
        )
        extra = cur.fetchall()
    conn.close()
    assert extra == [], (
        f"Unexpected non-PK indexes found: {extra}"
    )


@skip_if_no_db
def test_foreign_key_integrity_lineitem_orders():
    """Every l_orderkey in lineitem references an existing order."""
    from trackH_dbms.db import connect
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM lineitem l
            WHERE NOT EXISTS (
                SELECT 1 FROM orders o WHERE o.o_orderkey = l.l_orderkey
            )
            """
        )
        orphans = cur.fetchone()[0]
    conn.close()
    assert orphans == 0, f"{orphans} lineitem rows reference missing orders"


@skip_if_no_db
def test_foreign_key_integrity_orders_customer():
    """Every o_custkey in orders references an existing customer."""
    from trackH_dbms.db import connect
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM orders o
            WHERE NOT EXISTS (
                SELECT 1 FROM customer c WHERE c.c_custkey = o.o_custkey
            )
            """
        )
        orphans = cur.fetchone()[0]
    conn.close()
    assert orphans == 0, f"{orphans} orders rows reference missing customers"


@skip_if_no_db
def test_workload_queries_deterministic():
    """Running every workload query twice produces identical result checksums."""
    from trackH_dbms.db import connect
    from trackH_dbms.workload import QUERIES
    conn = connect()
    with conn.cursor() as cur:
        for q in QUERIES:
            cur.execute(q["sql"])
            rows1 = cur.fetchall()
            cur.execute(q["sql"])
            rows2 = cur.fetchall()
            md5_1 = _md5_rows(rows1)
            md5_2 = _md5_rows(rows2)
            assert md5_1 == md5_2, (
                f"Query {q['name']} returned different results on two runs"
            )
    conn.close()
