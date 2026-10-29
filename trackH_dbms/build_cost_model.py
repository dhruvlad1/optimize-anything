"""trackH_dbms/build_cost_model.py
--------------------------------
Builds the offline cost model for index selection on PostgreSQL:
1. Drops all existing 'optidx_%' secondary indexes and runs ANALYZE.
2. Measures planner estimated total cost for every workload query with no extra index (cost[q][none]).
3. For each candidate index in isolation:
   - Creates the index, records its creation time and size in bytes (pg_relation_size).
   - Runs ANALYZE on the indexed table.
   - Measures planner estimated total cost for every workload query (cost[q][c]).
   - Drops the index in a try/finally block so the database returns to baseline.
4. Computes additive workload benefit and benefit per MB for each candidate.
5. Saves results/trackH_cost_model.json and results/trackH_candidate_benefits.csv.

Usage:
  python -m trackH_dbms.build_cost_model

SAFETY INVARIANT:
Only indexes matching 'optidx_%' are ever created or dropped. Primary keys and
system catalogs are never touched.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
from typing import Any

from trackH_dbms.candidates import CANDIDATES, PREFIX
from trackH_dbms.db import connect
from trackH_dbms.workload import QUERIES

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def drop_all_optidx(conn) -> list[str]:
    """Drop any existing secondary indexes whose names begin with PREFIX.

    Returns the list of dropped index names. Primary keys are never touched.
    """
    dropped = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND indexname LIKE %s;
            """,
            (f"{PREFIX}%",),
        )
        rows = cur.fetchall()
        for (idx_name,) in rows:
            if idx_name.startswith(PREFIX):
                cur.execute(f"DROP INDEX IF EXISTS {idx_name};")
                dropped.append(idx_name)
    conn.commit()
    return dropped


def get_query_cost(conn, sql: str) -> float:
    """Return the planner's estimated total cost for *sql* using EXPLAIN FORMAT JSON."""
    explain_sql = f"EXPLAIN (FORMAT JSON) {sql.strip()}"
    with conn.cursor() as cur:
        cur.execute(explain_sql)
        plan_json = cur.fetchone()[0]
        return float(plan_json[0].get("Plan", {}).get("Total Cost", 0.0))


def build_cost_model() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect()
    print("Connected to PostgreSQL for cost model construction.")

    try:
        # Step 1: Clean slate and baseline costs
        dropped = drop_all_optidx(conn)
        if dropped:
            print(f"Cleaned up {len(dropped)} existing optidx indexes: {', '.join(dropped)}")

        print("Running ANALYZE across database for baseline...")
        with conn.cursor() as cur:
            cur.execute("ANALYZE;")
        conn.commit()

        print("Measuring baseline query costs (no secondary indexes)...")
        cost_none: dict[str, float] = {}
        for q in QUERIES:
            name = q["name"]
            c = get_query_cost(conn, q["sql"])
            cost_none[name] = round(c, 2)
            print(f"  {name:<30} baseline cost = {c:>10.2f}")

        # Step 2: Evaluate each candidate in isolation
        candidates_info: list[dict[str, Any]] = []
        sizes: dict[str, int] = {}
        cost_matrix: dict[str, dict[str, float]] = {q["name"]: {} for q in QUERIES}

        print(f"\nEvaluating {len(CANDIDATES)} candidate indexes one by one...")

        for idx, cand in enumerate(CANDIDATES, 1):
            name = cand["name"]
            table = cand["table"]
            create_sql = cand["sql"]
            rationale = cand["rationale"]

            if not name.startswith(PREFIX):
                raise RuntimeError(f"Safety violation: candidate name {name!r} does not start with {PREFIX!r}")

            print(f"[{idx}/{len(CANDIDATES)}] Testing {name} on {table}...")

            t_create = 0.0
            size_bytes = 0

            try:
                # Create candidate
                t0 = time.perf_counter()
                with conn.cursor() as cur:
                    cur.execute(create_sql)
                conn.commit()
                t_create = time.perf_counter() - t0

                # Analyze table and measure index size
                with conn.cursor() as cur:
                    cur.execute(f"ANALYZE {table};")
                    cur.execute("SELECT pg_relation_size(%s);", (name,))
                    size_bytes = int(cur.fetchone()[0])
                conn.commit()

                sizes[name] = size_bytes

                # Measure each query cost with this index active
                for q in QUERIES:
                    q_name = q["name"]
                    c_cost = get_query_cost(conn, q["sql"])
                    cost_matrix[q_name][name] = round(c_cost, 2)

            finally:
                # Guarantee index drop
                with conn.cursor() as cur:
                    cur.execute(f"DROP INDEX IF EXISTS {name};")
                conn.commit()

            # Additive benefit calculation
            total_benefit = 0.0
            for q in QUERIES:
                q_name = q["name"]
                f_q = q["weight"]
                c_none = cost_none[q_name]
                c_with = cost_matrix[q_name][name]
                # Benefit = weight * reduction in cost
                diff = c_none - c_with
                total_benefit += f_q * diff

            size_mb = size_bytes / (1024 * 1024)
            benefit_per_mb = (total_benefit / size_mb) if size_mb > 0 else 0.0

            candidates_info.append(
                {
                    "candidate": name,
                    "table": table,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_mb, 4),
                    "creation_time_s": round(t_create, 4),
                    "total_benefit": round(total_benefit, 2),
                    "benefit_per_mb": round(benefit_per_mb, 2),
                    "rationale": rationale,
                    "useless": cand.get("useless", False),
                }
            )

            print(
                f"       size = {size_mb:.2f} MB, build_time = {t_create:.3f}s, "
                f"total_benefit = {total_benefit:+.1f}, benefit/MB = {benefit_per_mb:+.1f}"
            )

        # Step 3: Write results/trackH_cost_model.json
        model_json_path = RESULTS_DIR / "trackH_cost_model.json"
        cost_model_data = {
            "queries": [q["name"] for q in QUERIES],
            "weights": {q["name"]: q["weight"] for q in QUERIES},
            "candidates": [c["name"] for c in CANDIDATES],
            "candidate_details": candidates_info,
            "sizes": sizes,
            "cost_none": cost_none,
            "cost_matrix": cost_matrix,
        }
        with open(model_json_path, "w", encoding="utf-8") as f:
            json.dump(cost_model_data, f, indent=2)
        print(f"\nCost model written to: {model_json_path}")

        # Step 4: Write CSV of candidate benefits
        csv_path = RESULTS_DIR / "trackH_candidate_benefits.csv"
        csv_alt_path = RESULTS_DIR / "trackH_cost_model.csv"
        fieldnames = [
            "candidate",
            "table",
            "size_bytes",
            "size_mb",
            "creation_time_s",
            "total_benefit",
            "benefit_per_mb",
            "rationale",
            "useless",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(candidates_info)
        with open(csv_alt_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(candidates_info)

        print(f"Per-candidate benefit CSV written to: {csv_path}")

    finally:
        # Final safety cleanup
        drop_all_optidx(conn)
        conn.close()
        print("Database connection closed and clean state verified.")


if __name__ == "__main__":
    try:
        build_cost_model()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
