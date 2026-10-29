"""trackH_dbms/run_index_selection.py
-----------------------------------
Runs storage-constrained index selection on PostgreSQL across budgets:
  B in {10%, 25%, 50%, 100%} of total candidate size.

Compares:
  - Greedy additive
  - DP additive (knapsack with sizes rounded UP to whole MB)
  - ILP assignment (OR-Tools SCIP)
  - Baseline (no secondary indexes)
  - All candidates (upper bound)

Measures:
  - Planner estimated cost (EXPLAIN FORMAT JSON)
  - Real execution latency (median of 5 runs after 1 warm-up)
  - MD5 result checksums (verifying exact semantic equivalence against baseline)
  - Model prediction error (model_cost vs real_est_cost)
  - Speedup vs baseline

Outputs:
  - results/trackH_index_selection.csv
  - results/trackH_plans_after/<q_name>.json (top 3 most improved queries)
  - results/trackH_latency_vs_budget.png

Usage:
  python -m trackH_dbms.run_index_selection
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import statistics
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from trackH_dbms.candidates import CANDIDATES, PREFIX
from trackH_dbms.db import connect
from trackH_dbms.index_solvers import (
    SolverResult,
    dp_additive,
    greedy_additive,
    ilp_assignment,
    model_cost,
)
from trackH_dbms.workload import QUERIES

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PLANS_AFTER_DIR = RESULTS_DIR / "trackH_plans_after"
N_TIMED_RUNS = 5


def _md5_rows(rows: list) -> str:
    """Compute MD5 checksum of sorted rows for deterministic result comparison."""
    h = hashlib.md5()
    for row in sorted(str(r) for r in rows):
        h.update(row.encode("utf-8"))
    return h.hexdigest()


def drop_all_optidx(conn) -> None:
    """Drop any active secondary index starting with optidx_. Never drops PKs."""
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
        for (idx_name,) in cur.fetchall():
            if idx_name.startswith(PREFIX):
                cur.execute(f"DROP INDEX IF EXISTS {idx_name};")
    conn.commit()


def run_index_selection() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_AFTER_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load offline cost model
    cost_model_path = RESULTS_DIR / "trackH_cost_model.json"
    if not cost_model_path.exists():
        raise FileNotFoundError(
            f"Cost model {cost_model_path} not found. "
            "Run 'python -m trackH_dbms.build_cost_model' first."
        )

    with open(cost_model_path, "r", encoding="utf-8") as f:
        cost_model = json.load(f)

    candidates: list[str] = cost_model["candidates"]
    sizes: dict[str, float] = {k: float(v) for k, v in cost_model["sizes"].items()}
    cost_matrix: dict[str, dict[str, float]] = cost_model["cost_matrix"]
    cost_none: dict[str, float] = {k: float(v) for k, v in cost_model["cost_none"].items()}
    weights: dict[str, float] = {k: float(v) for k, v in cost_model["weights"].items()}

    total_candidate_size = sum(sizes.values())
    total_candidate_size_mb = total_candidate_size / (1024 * 1024)
    print(f"Loaded cost model with {len(candidates)} candidates.")
    print(f"Total candidate storage: {total_candidate_size_mb:.2f} MB ({total_candidate_size:,} bytes)\n")

    # Map candidate name -> SQL statement & table
    cand_sql_map = {c["name"]: c["sql"] for c in CANDIDATES}
    cand_table_map = {c["name"]: c["table"] for c in CANDIDATES}

    # 2. Run solvers across budgets B in {10%, 25%, 50%, 100%}
    budgets = [0.10, 0.25, 0.50, 1.00]
    solver_runs: list[dict[str, Any]] = []

    print("=== Solving Index Selection Optimization Problem ===")
    print(f"{'Method':<12} {'Budget%':>8} {'Budget(MB)':>11} {'Size(MB)':>9} {'Model Cost':>12} {'Indexes Chosen'}")
    print("-" * 80)

    for frac in budgets:
        b_bytes = total_candidate_size * frac
        b_mb = b_bytes / (1024 * 1024)

        solvers = [
            ("greedy", greedy_additive(candidates, sizes, b_bytes, cost_matrix, cost_none, weights)),
            ("dp", dp_additive(candidates, sizes, b_bytes, cost_matrix, cost_none, weights)),
            ("ilp", ilp_assignment(candidates, sizes, b_bytes, cost_matrix, cost_none, weights)),
        ]

        for method_name, res in solvers:
            size_mb = res.total_size / (1024 * 1024)
            chosen_sorted = sorted(res.selected)
            print(
                f"{method_name:<12} {int(frac*100):>7}% {b_mb:>10.2f} {size_mb:>8.2f} "
                f"{res.model_cost:>12.2f} {', '.join(chosen_sorted) if chosen_sorted else '[]'}"
            )
            solver_runs.append(
                {
                    "method": method_name,
                    "budget_pct": int(frac * 100),
                    "budget_bytes": b_bytes,
                    "selected": chosen_sorted,
                    "model_size_bytes": res.total_size,
                    "model_size_mb": round(size_mb, 4),
                    "model_cost": res.model_cost,
                }
            )

    # 3. Formulate all configurations to test on live database
    # Configurations:
    #   - Baseline (no secondary indexes)
    #   - Each (method, budget) combination
    #   - All candidates
    configs_to_test: list[dict[str, Any]] = [
        {
            "method": "baseline",
            "budget_pct": 0,
            "selected": [],
            "model_size_mb": 0.0,
            "model_cost": model_cost(set(), cost_matrix, weights, cost_none),
        }
    ]
    configs_to_test.extend(solver_runs)
    configs_to_test.append(
        {
            "method": "all_candidates",
            "budget_pct": 100,
            "selected": sorted(candidates),
            "model_size_mb": round(total_candidate_size_mb, 4),
            "model_cost": model_cost(set(candidates), cost_matrix, weights, cost_none),
        }
    )

    # Connect to PostgreSQL
    conn = connect()
    print("\nConnected to PostgreSQL for physical measurement.")

    results_table: list[dict[str, Any]] = []
    baseline_weighted_lat = 0.0
    baseline_checksums: dict[str, str] = {}
    config_query_plans: dict[str, dict[str, dict]] = {}
    config_query_latencies: dict[str, dict[str, float]] = {}

    try:
        drop_all_optidx(conn)

        for cfg in configs_to_test:
            method = cfg["method"]
            bpct = cfg["budget_pct"]
            selected = cfg["selected"]
            model_sz_mb = cfg["model_size_mb"]
            m_cost = cfg["model_cost"]
            cfg_label = f"{method}_b{bpct}"

            print(f"\n--- Evaluating configuration: {method} (Budget: {bpct}%, {len(selected)} indexes) ---")

            # Try / finally block guarantees indexes are always dropped
            try:
                # 1. Create selected indexes
                tables_to_analyze = set()
                if selected:
                    with conn.cursor() as cur:
                        for idx_name in selected:
                            cur.execute(cand_sql_map[idx_name])
                            tables_to_analyze.add(cand_table_map[idx_name])
                    conn.commit()
                    # Run ANALYZE on affected tables
                    with conn.cursor() as cur:
                        for tbl in tables_to_analyze:
                            cur.execute(f"ANALYZE {tbl};")
                    conn.commit()

                # 2. Measure every workload query
                real_est_cost = 0.0
                total_weighted_lat = 0.0
                sum_weights = sum(weights.values())

                config_query_plans[cfg_label] = {}
                config_query_latencies[cfg_label] = {}
                checksums_match = True

                for q in QUERIES:
                    q_name = q["name"]
                    sql = q["sql"].strip()
                    f_q = weights[q_name]

                    # Explain plan and estimated cost
                    with conn.cursor() as cur:
                        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
                        plan_json = cur.fetchone()[0]

                    config_query_plans[cfg_label][q_name] = plan_json
                    q_est_cost = float(plan_json[0].get("Plan", {}).get("Total Cost", 0.0))
                    real_est_cost += f_q * q_est_cost

                    # 1 Warm-up run
                    with conn.cursor() as cur:
                        cur.execute(sql)
                        cur.fetchall()

                    # 5 Timed executions
                    latencies = []
                    last_rows = []
                    with conn.cursor() as cur:
                        for _ in range(N_TIMED_RUNS):
                            t0 = time.perf_counter()
                            cur.execute(sql)
                            last_rows = cur.fetchall()
                            t1 = time.perf_counter()
                            latencies.append((t1 - t0) * 1000.0)

                    median_lat = statistics.median(latencies)
                    config_query_latencies[cfg_label][q_name] = median_lat
                    total_weighted_lat += f_q * median_lat

                    # Checksum verification
                    current_chk = _md5_rows(last_rows)
                    if method == "baseline":
                        baseline_checksums[q_name] = current_chk
                    else:
                        if current_chk != baseline_checksums[q_name]:
                            checksums_match = False
                            raise RuntimeError(
                                f"RESULT INTEGRITY VIOLATION: Query {q_name} returned different "
                                f"rows under config {cfg_label} vs baseline!\n"
                                f"Baseline MD5: {baseline_checksums[q_name]}, Current MD5: {current_chk}"
                            )

                weighted_latency_ms = total_weighted_lat / sum_weights if sum_weights > 0 else 0.0

                if method == "baseline":
                    baseline_weighted_lat = weighted_latency_ms
                    speedup = 1.00
                else:
                    speedup = (baseline_weighted_lat / weighted_latency_ms) if weighted_latency_ms > 0 else 1.0

                # Prediction error: model_cost vs real_est_cost
                pred_error = m_cost - real_est_cost
                pred_error_pct = (pred_error / real_est_cost * 100.0) if real_est_cost > 0 else 0.0

                row_dict = {
                    "method": method,
                    "budget_pct": bpct,
                    "indexes": "|".join(selected) if selected else "none",
                    "total_size_mb": model_sz_mb,
                    "model_cost": round(m_cost, 2),
                    "real_est_cost": round(real_est_cost, 2),
                    "pred_error_pct": round(pred_error_pct, 2),
                    "weighted_latency_ms": round(weighted_latency_ms, 3),
                    "speedup_vs_baseline": round(speedup, 2),
                    "checksums_match": checksums_match,
                }
                results_table.append(row_dict)

                print(
                    f"  real_est_cost={real_est_cost:.1f}, latency={weighted_latency_ms:.2f}ms, "
                    f"speedup={speedup:.2f}x, pred_error={pred_error_pct:+.1f}%, checksums=OK"
                )

            finally:
                # Drop secondary indexes
                drop_all_optidx(conn)

    finally:
        drop_all_optidx(conn)
        conn.close()

    # 4. Save results/trackH_index_selection.csv
    df_results = pd.DataFrame(results_table)
    csv_out_path = RESULTS_DIR / "trackH_index_selection.csv"
    df_results.to_csv(csv_out_path, index=False)
    print(f"\nResults CSV written to {csv_out_path}")

    # 5. Identify best configuration and save top 3 improved query plans
    non_baseline = [r for r in results_table if r["method"] not in ("baseline", "all_candidates")]
    best_config_row = min(non_baseline, key=lambda r: r["weighted_latency_ms"]) if non_baseline else results_table[0]
    best_cfg_label = f"{best_config_row['method']}_b{best_config_row['budget_pct']}"

    print(f"\nBest configuration: {best_cfg_label} (Speedup: {best_config_row['speedup_vs_baseline']:.2f}x)")

    # Find top 3 queries with greatest latency reduction vs baseline
    baseline_label = "baseline_b0"
    query_improvements = []
    for q in QUERIES:
        qn = q["name"]
        b_lat = config_query_latencies[baseline_label][qn]
        opt_lat = config_query_latencies[best_cfg_label][qn]
        improvement_ms = b_lat - opt_lat
        query_improvements.append((qn, improvement_ms, b_lat, opt_lat))

    query_improvements.sort(key=lambda x: -x[1])
    top_3_improved = query_improvements[:3]

    print("\nTop 3 Most Improved Queries:")
    for qn, imp, b_lat, opt_lat in top_3_improved:
        print(f"  {qn}: {b_lat:.2f} ms -> {opt_lat:.2f} ms (saved {imp:.2f} ms)")
        # Save EXPLAIN JSON
        plan_json = config_query_plans[best_cfg_label][qn]
        plan_file = PLANS_AFTER_DIR / f"{qn}.json"
        with open(plan_file, "w", encoding="utf-8") as f:
            json.dump(plan_json, f, indent=2)
        print(f"    Saved plan to: {plan_file}")

    # 6. Print summary table
    print("\n=== Track H Index Selection Summary Table ===")
    display_cols = [
        "method", "budget_pct", "total_size_mb", "model_cost",
        "real_est_cost", "pred_error_pct", "weighted_latency_ms", "speedup_vs_baseline"
    ]
    print(df_results[display_cols].to_string(index=False))

    # 7. Generate Plot: weighted latency vs budget per method
    plt.figure(figsize=(9, 5))
    methods = ["greedy", "dp", "ilp"]
    markers = {"greedy": "o", "dp": "s", "ilp": "^"}
    colors = {"greedy": "#1f77b4", "dp": "#2ca02c", "ilp": "#d62728"}

    for m in methods:
        sub = df_results[df_results["method"] == m].sort_values("budget_pct")
        if not sub.empty:
            plt.plot(
                sub["budget_pct"],
                sub["weighted_latency_ms"],
                marker=markers.get(m, "o"),
                label=f"{m.upper()}",
                color=colors.get(m, "gray"),
                linewidth=2,
            )

    # Baseline horizontal line
    plt.axhline(
        y=baseline_weighted_lat,
        color="black",
        linestyle="--",
        label=f"Baseline (no indexes: {baseline_weighted_lat:.1f} ms)",
    )

    plt.xlabel("Storage Budget (% of Total Candidate Size)", fontsize=11)
    plt.ylabel("Workload Weighted Latency (ms)", fontsize=11)
    plt.title("Track H: Workload Latency vs Storage Budget by Selection Method", fontsize=12)
    plt.xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "trackH_latency_vs_budget.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    try:
        run_index_selection()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
