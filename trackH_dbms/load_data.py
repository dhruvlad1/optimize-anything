"""
trackH_dbms/load_data.py
------------------------
Generates a synthetic TPC-H-like dataset (seed=42) and bulk-loads it into
PostgreSQL via COPY.

Row counts (scale factor 0.1):
  region   :       5
  nation   :      25
  supplier :   1 000
  customer :  15 000
  part     :  20 000
  orders   : 150 000
  lineitem : ~600 000  (1–7 per order, average ~4)

Usage
-----
  python -m trackH_dbms.load_data

Requires PGPASSWORD to be set in the environment.
Drops and recreates all tables on every run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import sys
from datetime import date, timedelta

import numpy as np

from trackH_dbms.db import connect

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "trackH_dbms" / "data"
RESULTS_DIR = REPO_ROOT / "results"
SCHEMA_SQL = REPO_ROOT / "trackH_dbms" / "schema.sql"

# ---------------------------------------------------------------------------
# Row counts  (scale factor 0.1)
# ---------------------------------------------------------------------------
N_REGION = 5
N_NATION = 25
N_SUPPLIER = 1_000
N_CUSTOMER = 15_000
N_PART = 20_000
N_ORDERS = 150_000
LINES_PER_ORDER_MU = 4  # average line items per order (Poisson, clamped 1–7)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_MARKET_SEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"]
_ORDER_STATUS = ["O", "F", "P"]
_PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
_SHIP_MODES = ["AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB", "REG AIR"]
_SHIP_INSTR = ["DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN"]
_RETURN_FLAGS = ["N", "R", "A"]
_LINE_STATUS = ["O", "F"]
_CONTAINERS = ["SM BOX", "LG BOX", "MED BAG", "JUMBO BAG", "WRAP JAR"]
_PART_TYPES = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
_BRANDS = [f"Brand#{i}{j}" for i in range(1, 6) for j in range(1, 6)]


def _rand_date(start: date = date(1992, 1, 1), end: date = date(1998, 12, 31)) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(RNG.integers(0, delta)))


def _rand_str(length: int = 12) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz"
    return "".join(chars[i] for i in RNG.integers(0, len(chars), size=length))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def gen_region() -> list[tuple]:
    names = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
    return [(i, names[i], _rand_str(20)) for i in range(N_REGION)]


def gen_nation() -> list[tuple]:
    nation_names = [
        "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT",
        "ETHIOPIA", "FRANCE", "GERMANY", "INDIA", "INDONESIA",
        "IRAN", "IRAQ", "JAPAN", "JORDAN", "KENYA",
        "MOROCCO", "MOZAMBIQUE", "PERU", "CHINA", "ROMANIA",
        "SAUDI ARABIA", "VIETNAM", "RUSSIA", "UNITED KINGDOM", "UNITED STATES",
    ]
    return [
        (i, nation_names[i], int(RNG.integers(0, N_REGION)), _rand_str(20))
        for i in range(N_NATION)
    ]


def gen_supplier() -> list[tuple]:
    return [
        (
            i,
            f"Supplier#{i:09d}",
            _rand_str(25),
            int(RNG.integers(0, N_NATION)),
            f"{RNG.integers(10,99)}-{RNG.integers(100,999)}-{RNG.integers(1000,9999)}",
            round(float(RNG.uniform(-999.99, 9999.99)), 2),
            _rand_str(30),
        )
        for i in range(1, N_SUPPLIER + 1)
    ]


def gen_customer() -> list[tuple]:
    return [
        (
            i,
            f"Customer#{i:09d}",
            _rand_str(25),
            int(RNG.integers(0, N_NATION)),
            f"{RNG.integers(10,99)}-{RNG.integers(100,999)}-{RNG.integers(1000,9999)}",
            round(float(RNG.uniform(-999.99, 9999.99)), 2),
            _MARKET_SEGMENTS[int(RNG.integers(0, len(_MARKET_SEGMENTS)))],
            _rand_str(40),
        )
        for i in range(1, N_CUSTOMER + 1)
    ]


def gen_part() -> list[tuple]:
    return [
        (
            i,
            _rand_str(15),
            f"Manufacturer#{int(RNG.integers(1,6))}",
            _BRANDS[int(RNG.integers(0, len(_BRANDS)))],
            _PART_TYPES[int(RNG.integers(0, len(_PART_TYPES)))],
            int(RNG.integers(1, 51)),
            _CONTAINERS[int(RNG.integers(0, len(_CONTAINERS)))],
            round(float(RNG.uniform(1.00, 2000.00)), 2),
            _rand_str(10),
        )
        for i in range(1, N_PART + 1)
    ]


def gen_orders(customer_keys: list[int]) -> list[tuple]:
    rows = []
    for i in range(1, N_ORDERS + 1):
        odate = _rand_date()
        rows.append(
            (
                i,
                int(RNG.choice(customer_keys)),
                _ORDER_STATUS[int(RNG.integers(0, len(_ORDER_STATUS)))],
                round(float(RNG.uniform(1000.00, 500000.00)), 2),
                odate.isoformat(),
                _PRIORITIES[int(RNG.integers(0, len(_PRIORITIES)))],
                f"Clerk#{int(RNG.integers(1,1001)):09d}",
                0,
                _rand_str(25),
            )
        )
    return rows


def gen_lineitem(order_keys: list[int], part_keys: list[int], supp_keys: list[int]) -> list[tuple]:
    rows = []
    for okey in order_keys:
        n_lines = int(np.clip(RNG.poisson(LINES_PER_ORDER_MU), 1, 7))
        for lnum in range(1, n_lines + 1):
            sdate = _rand_date(date(1992, 1, 1), date(1998, 12, 1))
            cdate = _rand_date(sdate, date(1998, 12, 31))
            rdate = _rand_date(sdate, date(1998, 12, 31))
            qty = round(float(RNG.uniform(1, 50)), 2)
            price = round(float(RNG.uniform(1, 2000)), 2)
            disc = round(float(RNG.uniform(0, 0.1)), 2)
            tax = round(float(RNG.uniform(0, 0.08)), 2)
            rows.append(
                (
                    okey,
                    int(RNG.choice(part_keys)),
                    int(RNG.choice(supp_keys)),
                    lnum,
                    qty,
                    price,
                    disc,
                    tax,
                    _RETURN_FLAGS[int(RNG.integers(0, len(_RETURN_FLAGS)))],
                    _LINE_STATUS[int(RNG.integers(0, len(_LINE_STATUS)))],
                    sdate.isoformat(),
                    cdate.isoformat(),
                    rdate.isoformat(),
                    _SHIP_INSTR[int(RNG.integers(0, len(_SHIP_INSTR)))],
                    _SHIP_MODES[int(RNG.integers(0, len(_SHIP_MODES)))],
                    _rand_str(20),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(path: pathlib.Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _md5_file(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main load logic
# ---------------------------------------------------------------------------

def load_all() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic data …")
    regions = gen_region()
    nations = gen_nation()
    suppliers = gen_supplier()
    customers = gen_customer()
    parts = gen_part()
    orders = gen_orders([c[0] for c in customers])
    lineitems = gen_lineitem(
        [o[0] for o in orders],
        [p[0] for p in parts],
        [s[0] for s in suppliers],
    )

    tables = [
        ("region",   regions,    "region.csv"),
        ("nation",   nations,    "nation.csv"),
        ("supplier", suppliers,  "supplier.csv"),
        ("customer", customers,  "customer.csv"),
        ("part",     parts,      "part.csv"),
        ("orders",   orders,     "orders.csv"),
        ("lineitem", lineitems,  "lineitem.csv"),
    ]

    for _, rows, fname in tables:
        _write_csv(DATA_DIR / fname, rows)
    print(f"CSVs written to {DATA_DIR}")

    conn = connect()
    print("Connected to PostgreSQL.")

    with conn.cursor() as cur:
        # Apply schema (drops + creates)
        schema_sql = SCHEMA_SQL.read_text(encoding="utf-8")
        cur.execute(schema_sql)
        conn.commit()
        print("Schema applied.")

        # COPY each table
        for tname, _, fname in tables:
            csv_path = DATA_DIR / fname
            with open(csv_path, "r", encoding="utf-8") as f:
                with cur.copy(f"COPY {tname} FROM STDIN CSV") as copy:
                    copy.write(f.read())
            conn.commit()
            print(f"  Loaded {tname}")

        cur.execute("ANALYZE")
        conn.commit()
        print("ANALYZE complete.")

    # Summary
    summary = {}
    with conn.cursor() as cur:
        for tname, rows, fname in tables:
            cur.execute(f"SELECT COUNT(*) FROM {tname}")
            db_count = cur.fetchone()[0]
            cur.execute(
                "SELECT pg_total_relation_size(%s)",
                (tname,),
            )
            size_bytes = cur.fetchone()[0]
            summary[tname] = {
                "generated_rows": len(rows),
                "db_rows": db_count,
                "size_bytes": size_bytes,
                "csv_md5": _md5_file(DATA_DIR / fname),
            }

    conn.close()

    out_path = RESULTS_DIR / "trackH_data_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_path}")
    for t, s in summary.items():
        print(f"  {t:10s}  rows={s['db_rows']:6d}  size={s['size_bytes']:9d}B")


if __name__ == "__main__":
    try:
        load_all()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
