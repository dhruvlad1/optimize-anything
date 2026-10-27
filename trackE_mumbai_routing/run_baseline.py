"""run_baseline.py - Benchmark runner for Track E: DabbaDash (Mumbai CVRP).

Evaluates driver_order and nearest_neighbour baselines across customer sizes
n in [10, 20, 50, 100, 200] and seeds 0-4. Validates solutions, outputs
benchmark CSV, environment metadata JSON, a summary table, and route visualization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from trackE_mumbai_routing.baselines import driver_order, nearest_neighbour
from trackE_mumbai_routing.routing_common import (
    generate_instance,
    solution_cost,
    validate_solution,
)

N_SIZES: list[int] = [10, 20, 50, 100, 200]
SEEDS: list[int] = [0, 1, 2, 3, 4]
METHODS = [
    ("driver_order", driver_order),
    ("nearest_neighbour", nearest_neighbour),
]


def capture_environment_metadata(results_dir: Path) -> None:
    """Save execution environment metadata to JSON."""
    env_info = {
        "python_version": sys.version,
        "os_platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
        "cpu_count": os.cpu_count(),
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "pytest": pytest.__version__,
        },
    }
    json_path = results_dir / "trackE_environment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)
    print(f"Environment metadata written to: {json_path}")


def plot_baseline_map(instance, routes: list[list[int]], save_path: Path) -> None:
    """Plot nearest-neighbour vehicle routes on a 2D Mumbai map."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    coords = instance.coords  # [lat, lon]
    depot = coords[0]

    # Use a distinct colormap for routes
    cmap = plt.get_cmap("tab10")

    # Plot routes
    for r_idx, route in enumerate(routes):
        color = cmap(r_idx % 10)
        route_lons = [coords[node, 1] for node in route]
        route_lats = [coords[node, 0] for node in route]

        ax.plot(
            route_lons,
            route_lats,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=color,
            alpha=0.85,
            label=f"Route {r_idx + 1} ({len(route) - 2} stops)",
        )

    # Plot customer nodes
    cust_lons = coords[1:, 1]
    cust_lats = coords[1:, 0]
    ax.scatter(
        cust_lons,
        cust_lats,
        c="royalblue",
        s=30,
        zorder=4,
        edgecolors="black",
        linewidths=0.5,
        label="Customers",
    )

    # Plot depot with prominent marker
    ax.scatter(
        [depot[1]],
        [depot[0]],
        c="crimson",
        marker="*",
        s=250,
        zorder=5,
        edgecolors="black",
        linewidths=1.0,
        label="Depot (Andheri East)",
    )

    ax.set_title(
        f"Track E: DabbaDash Nearest-Neighbour Routes (n={instance.n}, seed={instance.seed})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Place legend neatly
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.28, 1.0),
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Map visualization saved to: {save_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    map_instance = None
    map_routes = None

    print("Running Track E (DabbaDash) baselines across n in [10, 20, 50, 100, 200] and seeds 0-4...\n")

    for n in N_SIZES:
        for seed in SEEDS:
            instance = generate_instance(n=n, seed=seed)
            dist = instance.dist_matrix
            assert dist is not None

            for method_name, method_func in METHODS:
                t0 = time.perf_counter()
                routes = method_func(instance)
                runtime_s = time.perf_counter() - t0

                # Validate solution independently
                is_valid, errors = validate_solution(instance, routes)
                if not is_valid:
                    raise RuntimeError(
                        f"Validation FAILED for method='{method_name}', n={n}, seed={seed}!\n"
                        f"Errors:\n" + "\n".join(errors)
                    )

                cost_km = solution_cost(routes, dist)
                num_vehicles = len(routes)

                records.append({
                    "method": method_name,
                    "n": n,
                    "seed": seed,
                    "distance_km": round(cost_km, 3),
                    "vehicles": num_vehicles,
                    "runtime_s": round(runtime_s, 6),
                    "valid": is_valid,
                })

                # Capture instance and routes for map plotting (n=50, seed=0, nearest_neighbour)
                if n == 50 and seed == 0 and method_name == "nearest_neighbour":
                    map_instance = instance
                    map_routes = routes

    df = pd.DataFrame(records)
    csv_path = results_dir / "trackE_baseline.csv"
    df.to_csv(csv_path, index=False)
    print(f"Benchmark results written to: {csv_path}\n")

    # Save environment metadata
    capture_environment_metadata(results_dir)

    # Save visualization map
    if map_instance is not None and map_routes is not None:
        map_path = results_dir / "trackE_baseline_map.png"
        plot_baseline_map(map_instance, map_routes, map_path)

    # Summary table: mean +/- std distance per method and n
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Mean +/- Std Distance (km) per Method and Problem Size (n)")
    print("=" * 70)

    summary = (
        df.groupby(["n", "method"])["distance_km"]
        .agg(["mean", "std"])
        .reset_index()
    )

    pivot_mean = df.pivot_table(
        index="n", columns="method", values="distance_km", aggfunc="mean"
    )
    pivot_std = df.pivot_table(
        index="n", columns="method", values="distance_km", aggfunc="std"
    )

    header = f"{'n':<8} | {'driver_order (km)':<25} | {'nearest_neighbour (km)':<25} | {'improvement (%)':<15}"
    print(header)
    print("-" * len(header))

    for n in N_SIZES:
        do_mean = pivot_mean.loc[n, "driver_order"]
        do_std = pivot_std.loc[n, "driver_order"]
        nn_mean = pivot_mean.loc[n, "nearest_neighbour"]
        nn_std = pivot_std.loc[n, "nearest_neighbour"]
        impr = ((do_mean - nn_mean) / do_mean) * 100.0

        do_str = f"{do_mean:.2f} +/- {do_std:.2f}"
        nn_str = f"{nn_mean:.2f} +/- {nn_std:.2f}"
        impr_str = f"{impr:+.1f}%"

        print(f"{n:<8} | {do_str:<25} | {nn_str:<25} | {impr_str:<15}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
