"""
trackH_dbms/run_baseline.py
---------------------------
Runs the workload defined in workload.py against the database and records
baseline (no secondary indexes) performance metrics.

Outputs
-------
  results/trackH_baseline.csv              — per-query summary
  results/trackH_baseline_plans/<name>.json — EXPLAIN ANALYZE JSON plans
  results/trackH_environment.json          — PostgreSQL version + key settings

Usage
-----
  python trackH_dbms/run_baseline.py

Requires PGPASSWORD to be set. Run load_data.py first to populate the DB.
"""

from __future__ import annotations

import csv
import json
import pathlib
import statistics
import sys
import time

from trackH_dbms.db import connect
from trackH_dbms.workload import QUERIES

REPO_ROOT = pathlib.Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PLANS_DIR = RESULTS_DIR / "trackH_baseline_plans"
N_TIMED_RUNS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_plan_summary(plan_json: list[dict]) -> dict:
    """Extract key metrics from EXPLAIN ANALYZE JSON output."""
    root = plan_json[0]
    top_plan = root.get("Plan", {})

    def _collect_nodes(node: dict, acc: list) -> None:
        acc.append(node.get("Node Type", "Unknown"))
        for child in node.get("Plans", []):
            _collect_nodes(child, acc)

    nodes: list[str] = []
    _collect_nodes(top_plan, nodes)

    return {
        "planning_time_ms": root.get("Planning Time", 0.0),
        "execution_time_ms": root.get("Execution Time", 0.0),
        "estimated_total_cost": top_plan.get("Total Cost", 0.0),
        "shared_hit_blocks": top_plan.get("Shared Hit Blocks", 0),
        "shared_read_blocks": top_plan.get("Shared Read Blocks", 0),
        "node_types": nodes,
    }


def _time_query(conn, sql: str, n_runs: int = N_TIMED_RUNS) -> tuple[float, int]:
    """Run *sql* n_runs times; return (median_ms, row_count)."""
    latencies = []
    row_count = 0
    with conn.cursor() as cur:
        for _ in range(n_runs):
            t0 = time.perf_counter()
            cur.execute(sql)
            rows = cur.fetchall()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1_000)
            row_count = len(rows)
    return statistics.median(latencies), row_count


def _pg_environment(conn) -> dict:
    """Capture PostgreSQL version and key settings."""
    env = {}
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        env["pg_version"] = cur.fetchone()[0]

        interesting = {
            "shared_buffers", "work_mem", "effective_cache_size",
            "max_connections", "random_page_cost", "seq_page_cost",
            "default_statistics_target",
        }
        cur.execute("SHOW ALL")
        rows = cur.fetchall()
        for name, value, *_ in rows:
            if name in interesting:
                env[name] = value
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_DIR.mkdir(parents=True, exist_ok=True)

    conn = connect()
    print("Connected to PostgreSQL.\n")

    # Capture environment
    pg_env = _pg_environment(conn)
    env_path = RESULTS_DIR / "trackH_environment.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(pg_env, f, indent=2)
    print(f"Environment info written to {env_path}")

    summary_rows = []
    total_weighted_ms = 0.0
    total_weight = 0

    print(f"\n{'Query':<35} {'Median ms':>10} {'Rows':>8} {'Est. cost':>12}")
    print("-" * 70)

    for q in QUERIES:
        name = q["name"]
        sql = q["sql"].strip()
        weight = q["weight"]

        # EXPLAIN ANALYZE
        explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
        with conn.cursor() as cur:
            cur.execute(explain_sql)
            plan_json = cur.fetchone()[0]

        plan_summary = _extract_plan_summary(plan_json)

        # Save plan JSON
        plan_path = PLANS_DIR / f"{name}.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_json, f, indent=2)

        # Warm-up run (not timed)
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.fetchall()

        # Timed runs
        median_ms, row_count = _time_query(conn, sql)

        total_weighted_ms += weight * median_ms
        total_weight += weight

        print(f"{name:<35} {median_ms:>10.2f} {row_count:>8} {plan_summary['estimated_total_cost']:>12.1f}")

        summary_rows.append(
            {
                "query": name,
                "weight": weight,
                "median_latency_ms": round(median_ms, 3),
                "row_count": row_count,
                "planning_time_ms": round(plan_summary["planning_time_ms"], 3),
                "execution_time_ms": round(plan_summary["execution_time_ms"], 3),
                "estimated_total_cost": plan_summary["estimated_total_cost"],
                "shared_hit_blocks": plan_summary["shared_hit_blocks"],
                "shared_read_blocks": plan_summary["shared_read_blocks"],
                "plan_node_types": "|".join(plan_summary["node_types"]),
            }
        )

    conn.close()

    # Write CSV
    csv_path = RESULTS_DIR / "trackH_baseline.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    weighted_avg = total_weighted_ms / total_weight if total_weight else 0.0
    print("-" * 70)
    print(f"Weighted average latency: {weighted_avg:.2f} ms")
    print(f"\nBaseline CSV  : {csv_path}")
    print(f"Plans dir     : {PLANS_DIR}")


if __name__ == "__main__":
    try:
        run_baseline()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
