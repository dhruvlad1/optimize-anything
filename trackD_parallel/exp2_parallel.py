"""trackD_parallel/exp2_parallel.py
----------------------------------
Experiment 2: Parallel Computing Benchmarks
- (a) Embarrassingly parallel scaling & Amdahl/Gustafson fitting (N = 48 tasks, ~0.5s each)
- (b) Parallel overhead check on tiny tasks (~5ms) & crossover duration
- (c) Load balancing on heavy-tailed tasks (Static equal split, Static LPT, Dynamic queue)
- (d) Work-stealing discrete-event SIMULATION across scaling worker pools
- (e) Data parallelism: Persistent-worker sharded logistic regression gradient descent
- (f) Pipeline parallelism: 3-stage queue-connected pipeline vs sequential processing
- (g) Model parallelism: Column-sharded multithreaded matrix activation y = tanh(X W)

Saves:
  results/trackD_parallel.csv
  results/trackD_parallel_scaling.png
"""

from __future__ import annotations

# ===========================================================================
# Step 0: Set BLAS threads to 1 BEFORE numpy is imported anywhere
# ===========================================================================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import multiprocessing as mp
import pathlib
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from trackD_parallel.parallel_ops import (
    amdahl_speedup,
    batch_load_balance_task,
    calibrate_cpu_task,
    cpu_bound_task,
    data_parallel_worker,
    fit_amdahl,
    fit_gustafson,
    gustafson_speedup,
    karp_flatt_serial_fraction,
    load_balance_task,
    lpt_schedule,
    pipeline_stage_worker,
    run_cpu_task_wrapper,
    simulate_dynamic_schedule,
    simulate_static_schedule,
    simulate_work_stealing,
)

RESULTS_DIR = pathlib.Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 2: Parallel Computing Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced reps and task counts")
    return parser.parse_args()


# ===========================================================================
# Main Benchmark Driver
# ===========================================================================

def main():
    print("[CONFIG] BLAS thread limits set: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1", flush=True)
    args = parse_args()
    print("=" * 80)
    print("TRACK D: EXPERIMENT 2 - PARALLEL COMPUTING BENCHMARKS")
    if args.quick:
        print("MODE: --quick (1 repetition, reduced task counts)")
    print("=" * 80, flush=True)

    # Hardware detection
    logical_cpus = os.cpu_count() or 1
    physical_cpus = psutil.cpu_count(logical=False) if hasattr(psutil, "cpu_count") else logical_cpus
    print(f"[HARDWARE] Logical CPUs: {logical_cpus}, Physical Cores: {physical_cpus}", flush=True)

    # Process pool configurations to test
    max_procs = min(16, logical_cpus)
    procs_candidates = [1, 2, 4, 8, max_procs]
    # Remove duplicates and sort
    procs_list = sorted(list(set(p for p in procs_candidates if p <= max_procs)))
    print(f"[CONFIG] Worker process configurations to evaluate: {procs_list}", flush=True)

    # -----------------------------------------------------------------------
    # Task Calibration and Runtime Estimation
    # -----------------------------------------------------------------------
    print("\n--- Calibrating CPU-bound task to ~0.50 s ---", flush=True)
    calib_reps_05 = calibrate_cpu_task(target_seconds=0.50, dim=250)
    measured_task_05 = cpu_bound_task(calib_reps_05, dim=250)
    if abs(measured_task_05 - 0.50) > 0.08 and measured_task_05 > 0:
        calib_reps_05 = max(1, int(calib_reps_05 * (0.50 / measured_task_05)))
        measured_task_05 = cpu_bound_task(calib_reps_05, dim=250)
    print(f"  Calibrated 0.5s task: reps={calib_reps_05}, measured time={measured_task_05:.4f}s [MEASURED]", flush=True)

    calib_reps_5ms = calibrate_cpu_task(target_seconds=0.005, dim=150)
    measured_task_5ms = cpu_bound_task(calib_reps_5ms, dim=150)
    print(f"  Calibrated 5ms task: reps={calib_reps_5ms}, measured time={measured_task_5ms*1e3:.2f}ms [MEASURED]", flush=True)

    # Runtime Estimation
    reps_per_config = 1 if args.quick else 3
    n_tasks_a = 24 if args.quick else 48
    est_a = sum(reps_per_config * max(measured_task_05, (n_tasks_a * measured_task_05) / p) for p in procs_list)
    est_gustafson = sum(reps_per_config * 4 * measured_task_05 for _ in procs_list)
    est_b = 3.0
    est_c = 20.0 if args.quick else 55.0
    est_e = 3.0
    est_f = 15.0
    est_g = 3.0
    total_est_s = est_a + est_gustafson + est_b + est_c + est_e + est_f + est_g

    print("\n" + "=" * 60)
    print("RUNTIME ESTIMATE: exp2_parallel")
    print("=" * 60)
    print(f"  Part (a) Scaling + Gustafson:  ~{est_a + est_gustafson:.1f} s")
    print(f"  Part (b) Overhead & Crossover: ~{est_b:.1f} s")
    print(f"  Part (c) Load Balancing:       ~{est_c:.1f} s")
    print(f"  Part (d) Work Stealing Sim:    ~0.1 s")
    print(f"  Part (e) Data Parallelism:     ~{est_e:.1f} s")
    print(f"  Part (f) Pipeline:             ~{est_f:.1f} s")
    print(f"  Part (g) Model Parallelism:    ~{est_g:.1f} s")
    print(f"  Estimated Total Runtime:       {total_est_s:.1f} s ({total_est_s/60:.2f} min)")
    print("=" * 60, flush=True)

    if total_est_s > 600 and not args.quick:
        print("[WARNING] Estimated runtime > 10 min, reducing repetitions from 3 to 2.")
        reps_per_config = 2

    records_csv: List[Dict[str, Any]] = []

    # =======================================================================
    # Part (a): Embarrassingly Parallel Scaling & Amdahl / Gustafson Fitting
    # =======================================================================
    print("\n" + "=" * 80)
    print(f"PART (A): Embarrassingly Parallel Scaling (N = {n_tasks_a} tasks, ~0.5s each)")
    print("=" * 80, flush=True)

    part_a_results: List[Dict[str, Any]] = []
    t_seq_baseline = None

    for p in procs_list:
        # Measure pool start-up time outside timed region
        t_pool_start = time.perf_counter()
        pool = mp.Pool(processes=p)
        # Warm up workers outside timed region
        pool.map(run_cpu_task_wrapper, [1] * p)
        pool_startup_time = time.perf_counter() - t_pool_start

        durations_reps: List[float] = []
        for rep in range(reps_per_config):
            t_run_start = time.perf_counter()
            _ = pool.map(run_cpu_task_wrapper, [calib_reps_05] * n_tasks_a)
            durations_reps.append(time.perf_counter() - t_run_start)

        pool.close()
        pool.join()

        mean_dur = float(np.mean(durations_reps))
        if p == 1:
            t_seq_baseline = mean_dur

        speedup = t_seq_baseline / mean_dur
        efficiency = speedup / p
        kf_serial = karp_flatt_serial_fraction(p, speedup) if p > 1 else 0.0

        print(
            f"  p={p:2d}: Wall Time={mean_dur:.3f}s [MEASURED] | Speedup={speedup:.2f}x [MEASURED] | "
            f"Efficiency={efficiency:.2f} [MEASURED] | Karp-Flatt e={kf_serial:.4f} [MEASURED] | "
            f"Pool Startup={pool_startup_time*1e3:.1f}ms",
            flush=True,
        )

        part_a_results.append({
            "procs": p,
            "wall_time_s": mean_dur,
            "speedup": speedup,
            "efficiency": efficiency,
            "karp_flatt_serial": kf_serial,
            "pool_startup_ms": pool_startup_time * 1e3,
        })
        records_csv.append({
            "experiment": "part_a_strong_scaling",
            "procs": p,
            "metric": "speedup",
            "value": speedup,
            "type": "measured",
        })

    # Fit Amdahl's Law
    procs_arr = [r["procs"] for r in part_a_results]
    speedups_arr = [r["speedup"] for r in part_a_results]
    s_amdahl = fit_amdahl(procs_arr, speedups_arr)
    print(f"\n  Fitted Amdahl Serial Fraction s = {s_amdahl:.4f} [MEASURED FIT] (Predicted asymptotic max speedup = {1.0/max(1e-6, s_amdahl):.2f}x)", flush=True)

    # Gustafson's Weak Scaling (scaled workload: tasks proportional to p)
    print("\n  Evaluating Gustafson's Scaled Workload (Weak Scaling, 4 tasks per worker)...", flush=True)
    gustafson_results: List[Dict[str, Any]] = []
    base_tasks_per_proc = 4

    # Baseline sequential time for 4 tasks
    pool1 = mp.Pool(processes=1)
    pool1.map(run_cpu_task_wrapper, [1])
    t0 = time.perf_counter()
    _ = pool1.map(run_cpu_task_wrapper, [calib_reps_05] * base_tasks_per_proc)
    t1_base = time.perf_counter() - t0
    pool1.close()
    pool1.join()

    for p in procs_list:
        n_scaled_tasks = base_tasks_per_proc * p
        pool = mp.Pool(processes=p)
        pool.map(run_cpu_task_wrapper, [1] * p)
        t_start = time.perf_counter()
        _ = pool.map(run_cpu_task_wrapper, [calib_reps_05] * n_scaled_tasks)
        t_p_scaled = time.perf_counter() - t_start
        pool.close()
        pool.join()

        # Scaled speedup = (sequential time for n_scaled_tasks) / (parallel time)
        # = (p * t1_base) / t_p_scaled
        scaled_speedup = (p * t1_base) / t_p_scaled
        print(f"  p={p:2d}: Workload={n_scaled_tasks:2d} tasks | Parallel Time={t_p_scaled:.3f}s [MEASURED] | Scaled Speedup={scaled_speedup:.2f}x [MEASURED]", flush=True)
        gustafson_results.append({"procs": p, "scaled_speedup": scaled_speedup})
        records_csv.append({
            "experiment": "part_a_gustafson_scaling",
            "procs": p,
            "metric": "scaled_speedup",
            "value": scaled_speedup,
            "type": "measured",
        })

    s_gustafson = fit_gustafson([r["procs"] for r in gustafson_results], [r["scaled_speedup"] for r in gustafson_results])
    print(f"  Fitted Gustafson Serial Fraction s = {s_gustafson:.4f} [MEASURED FIT]", flush=True)

    # =======================================================================
    # Part (b): Parallel Overhead Check on Tiny Tasks (~5ms) & Crossover
    # =======================================================================
    print("\n" + "=" * 80)
    print("PART (B): Parallel Overhead Check on Tiny Tasks (~5 ms) & Crossover Search")
    print("=" * 80, flush=True)

    tiny_tasks_count = 48
    # 1. Sequential baseline
    t0 = time.perf_counter()
    for _ in range(tiny_tasks_count):
        cpu_bound_task(calib_reps_5ms, dim=150)
    t_tiny_seq = time.perf_counter() - t0

    # 2. Parallel with 4 processes (pool lifecycle overhead included)
    p_tiny = min(4, max_procs)
    t0 = time.perf_counter()
    with mp.Pool(processes=p_tiny) as pool_tiny:
        _ = pool_tiny.map(run_cpu_task_wrapper, [(calib_reps_5ms, 150)] * tiny_tasks_count)
    t_tiny_par = time.perf_counter() - t0

    speedup_tiny = t_tiny_seq / t_tiny_par
    print(f"  N={tiny_tasks_count} tiny tasks (~5ms each):")
    print(f"    Sequential Time: {t_tiny_seq*1e3:.2f} ms [MEASURED]")
    print(f"    Parallel Time (p={p_tiny}, fresh pool): {t_tiny_par*1e3:.2f} ms [MEASURED]")
    print(f"    Speedup: {speedup_tiny:.2f}x [MEASURED] (Speedup < 1 confirms process creation & IPC overhead dominates)")

    # Steady-state check with pre-warmed pool
    pool_warmed = mp.Pool(processes=p_tiny)
    pool_warmed.map(run_cpu_task_wrapper, [(1, 150)] * p_tiny)
    t0 = time.perf_counter()
    _ = pool_warmed.map(run_cpu_task_wrapper, [(calib_reps_5ms, 150)] * tiny_tasks_count)
    t_tiny_warmed = time.perf_counter() - t0
    pool_warmed.close()
    pool_warmed.join()
    print(f"    Parallel Time (p={p_tiny}, warmed pool): {t_tiny_warmed*1e3:.2f} ms (Speedup: {t_tiny_seq/t_tiny_warmed:.2f}x) [MEASURED]")

    # Crossover search: vary task duration from 5ms to 120ms with fresh pool
    crossover_dur_s: Optional[float] = None
    crossover_table: List[Tuple[float, float, float, float]] = []
    print("\n  Searching for crossover task duration (speedup reaches 1.0 on p=4):", flush=True)

    test_durations_ms = [5.0, 15.0, 30.0, 50.0, 80.0, 120.0]
    for dur_ms in test_durations_ms:
        r = calibrate_cpu_task(target_seconds=dur_ms / 1000.0, dim=150)
        # Sequential
        t0 = time.perf_counter()
        for _ in range(24):
            cpu_bound_task(r, dim=150)
        t_s = time.perf_counter() - t0

        # Parallel p=4 (pool lifecycle included)
        t0 = time.perf_counter()
        with mp.Pool(processes=p_tiny) as p_cross:
            _ = p_cross.map(run_cpu_task_wrapper, [(r, 150)] * 24)
        t_p = time.perf_counter() - t0

        sp = t_s / t_p
        crossover_table.append((dur_ms, t_s * 1e3, t_p * 1e3, sp))
        print(f"    Task ~{dur_ms:4.1f} ms -> Seq={t_s*1e3:6.1f} ms, Par={t_p*1e3:6.1f} ms, Speedup={sp:.2f}x [MEASURED]", flush=True)
        if sp >= 1.0 and crossover_dur_s is None:
            crossover_dur_s = dur_ms / 1000.0

    if crossover_dur_s is not None:
        print(f"  Observed Crossover Task Duration: ~{crossover_dur_s*1e3:.1f} ms [MEASURED]", flush=True)
    else:
        print("  Crossover occurs at > 120 ms on this platform [MEASURED]", flush=True)

    # =======================================================================
    # Part (c): Load Balancing on Heavy-Tailed Tasks
    # =======================================================================
    print("\n" + "=" * 80)
    print("PART (C): Load Balancing on Heavy-Tailed Tasks (Static vs LPT vs Dynamic)")
    print("=" * 80, flush=True)

    n_tasks_c = 48 if args.quick else 96
    p_workers = min(4, max_procs)

    # Generate lognormal task durations (mean ~0.3s, some 5-10x longer)
    rng_lb = np.random.default_rng(42)
    # lognormal with mean ~0.3
    raw_samples = rng_lb.lognormal(mean=-1.2, sigma=0.75, size=n_tasks_c)
    # Scale so mean duration is ~0.25 - 0.30s
    scale_factor = 0.28 / np.mean(raw_samples)
    target_durations = raw_samples * scale_factor
    # Convert each target duration to calibrated reps
    base_dim = 180
    base_rate = calibrate_cpu_task(target_seconds=0.1, dim=base_dim) / 0.1
    task_reps = [max(1, int(d * base_rate)) for d in target_durations]

    print(f"  Generated {n_tasks_c} heavy-tailed tasks: min={min(target_durations):.3f}s, mean={np.mean(target_durations):.3f}s, max={max(target_durations):.3f}s", flush=True)

    # Sequential 1-process baseline
    print("  Measuring sequential baseline (1 worker)...", flush=True)
    t0 = time.perf_counter()
    measured_task_durations: List[float] = []
    for idx, r in enumerate(task_reps):
        t_d = cpu_bound_task(r, dim=base_dim)
        measured_task_durations.append(t_d)
    t_seq_heavy = time.perf_counter() - t0
    print(f"  Sequential total work: {t_seq_heavy:.3f} s [MEASURED]", flush=True)

    lb_strategies = ["Static Equal Split", "Static LPT", "Dynamic Queue"]
    lb_summary: List[Dict[str, Any]] = []

    for strat in lb_strategies:
        makespans_reps: List[float] = []
        imbalances_reps: List[float] = []

        for rep in range(reps_per_config):
            pool = mp.Pool(processes=p_workers)

            if strat == "Static Equal Split":
                # Contiguous blocks: partition indices into p_workers chunks
                chunk_size = int(np.ceil(n_tasks_c / p_workers))
                batches = []
                for w in range(p_workers):
                    batch_tasks = [
                        (idx, task_reps[idx], base_dim)
                        for idx in range(w * chunk_size, min((w + 1) * chunk_size, n_tasks_c))
                    ]
                    batches.append(batch_tasks)

                t0 = time.perf_counter()
                results = pool.map(batch_load_balance_task, batches)
                dur = time.perf_counter() - t0
                worker_times = [r[1] for r in results]

            elif strat == "Static LPT":
                # Longest Processing Time assignment
                lpt_indices = lpt_schedule(measured_task_durations, p_workers)
                batches = []
                for w in range(p_workers):
                    batch_tasks = [(idx, task_reps[idx], base_dim) for idx in lpt_indices[w]]
                    batches.append(batch_tasks)

                t0 = time.perf_counter()
                results = pool.map(batch_load_balance_task, batches)
                dur = time.perf_counter() - t0
                worker_times = [r[1] for r in results]

            elif strat == "Dynamic Queue":
                # Dynamic shared queue via imap_unordered with chunksize=1
                task_args = [(idx, task_reps[idx], base_dim) for idx in range(n_tasks_c)]
                t0 = time.perf_counter()
                results_dyn = list(pool.imap_unordered(load_balance_task, task_args, chunksize=1))
                dur = time.perf_counter() - t0

                # Group elapsed times by worker PID
                pid_totals: Dict[int, float] = {}
                for task_id, pid, el in results_dyn:
                    pid_totals[pid] = pid_totals.get(pid, 0.0) + el
                worker_times = list(pid_totals.values())

            pool.close()
            pool.join()

            makespans_reps.append(dur)
            imbalance = max(worker_times) / max(1e-6, np.mean(worker_times))
            imbalances_reps.append(imbalance)

        mean_ms = float(np.mean(makespans_reps))
        mean_imb = float(np.mean(imbalances_reps))
        sp = t_seq_heavy / mean_ms

        print(
            f"  {strat:18s}: Makespan={mean_ms:.3f}s [MEASURED] | Speedup={sp:.2f}x [MEASURED] | "
            f"Imbalance (max/mean)={mean_imb:.2f} [MEASURED]",
            flush=True,
        )

        lb_summary.append({
            "strategy": strat,
            "makespan_s": mean_ms,
            "speedup": sp,
            "imbalance": mean_imb,
        })
        records_csv.append({
            "experiment": "part_c_load_balancing",
            "strategy": strat,
            "makespan_s": mean_ms,
            "speedup": sp,
            "imbalance": mean_imb,
            "type": "measured",
        })

    # =======================================================================
    # Part (d): Work-Stealing Discrete-Event SIMULATION
    # =======================================================================
    print("\n" + "=" * 80)
    print("PART (D): Work-Stealing Discrete-Event SIMULATION (Simulated)")
    print("=" * 80, flush=True)

    sim_workers_list = [2, 4, 8, 16, 32]
    sim_results: List[Dict[str, Any]] = []

    print("  Workers | Static Makespan | Dynamic Makespan | Work-Stealing Makespan | Steals Count | Lower Bound")
    print("  " + "-" * 88)

    for p in sim_workers_list:
        stat_m = simulate_static_schedule(measured_task_durations, p)
        dyn_m = simulate_dynamic_schedule(measured_task_durations, p)
        ws_res = simulate_work_stealing(measured_task_durations, p)

        ws_m = ws_res["simulated_makespan"]
        steals = ws_res["num_steals"]
        lb = ws_res["theoretical_lower_bound"]

        print(
            f"    p={p:2d}   |    {stat_m:6.3f} s    |    {dyn_m:6.3f} s    |       {ws_m:6.3f} s        |     {steals:4d}     |   {lb:6.3f} s [SIMULATED]",
            flush=True,
        )

        sim_results.append({
            "procs": p,
            "static_sim_s": stat_m,
            "dynamic_sim_s": dyn_m,
            "work_stealing_sim_s": ws_m,
            "steals": steals,
            "lower_bound_s": lb,
        })
        records_csv.append({
            "experiment": "part_d_work_stealing_simulation",
            "procs": p,
            "sim_makespan_s": ws_m,
            "steals": steals,
            "type": "simulated",
        })

    # Sanity check: compare p=4 simulation with measured from Part (c)
    meas_stat_c = [r["makespan_s"] for r in lb_summary if r["strategy"] == "Static Equal Split"][0]
    meas_dyn_c = [r["makespan_s"] for r in lb_summary if r["strategy"] == "Dynamic Queue"][0]
    sim_4 = [r for r in sim_results if r["procs"] == 4][0]

    print(f"\n  Sanity Check Comparison at p=4:")
    print(f"    Static:  Measured={meas_stat_c:.3f} s vs Simulated={sim_4['static_sim_s']:.3f} s")
    print(f"    Dynamic: Measured={meas_dyn_c:.3f} s vs Simulated={sim_4['dynamic_sim_s']:.3f} s")

    # =======================================================================
    # Part (e): Data Parallelism: Logistic Regression Sharded Gradient Descent
    # =======================================================================
    print("\n" + "=" * 80)
    print("PART (E): Data Parallelism (Logistic Regression on 200,000 samples, 50 features)")
    print("=" * 80, flush=True)

    n_samples, n_features = 200000, 50
    rng_lr = np.random.default_rng(0)
    X = rng_lr.standard_normal((n_samples, n_features), dtype=np.float64)
    true_w = rng_lr.standard_normal(n_features)
    prob = 1.0 / (1.0 + np.exp(-np.clip(X @ true_w, -30.0, 30.0)))
    y = (rng_lr.uniform(0.0, 1.0, n_samples) < prob).astype(np.float64)

    n_gd_iters = 20
    learning_rate = 0.1

    # Sequential baseline
    print("  Running sequential logistic regression (p=1)...", flush=True)
    w_seq = np.zeros(n_features, dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(n_gd_iters):
        z = X @ w_seq
        p_hat = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        grad = (X.T @ (p_hat - y)) / n_samples
        w_seq -= learning_rate * grad
    t_seq_lr = time.perf_counter() - t0
    ms_per_iter_seq = (t_seq_lr / n_gd_iters) * 1e3

    # Final sequential loss (binary cross-entropy)
    z_final = X @ w_seq
    p_final = 1.0 / (1.0 + np.exp(-np.clip(z_final, -30.0, 30.0)))
    seq_loss = -float(np.mean(y * np.log(np.clip(p_final, 1e-15, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p_final, 1e-15, 1.0))))

    print(f"  Sequential: Total Time={t_seq_lr*1e3:.1f}ms, Time/Iter={ms_per_iter_seq:.2f}ms [MEASURED], Final Loss={seq_loss:.8f}")

    data_par_records: List[Dict[str, Any]] = []

    for p_procs in [1, 2, 4]:
        # Persistent workers sharded once
        pipes: List[Any] = []
        processes: List[mp.Process] = []

        shard_size = n_samples // p_procs
        for i in range(p_procs):
            parent_conn, child_conn = mp.Pipe()
            X_shard = X[i * shard_size : (i + 1) * shard_size]
            y_shard = y[i * shard_size : (i + 1) * shard_size]
            proc = mp.Process(target=data_parallel_worker, args=(child_conn, X_shard, y_shard))
            proc.start()
            pipes.append(parent_conn)
            processes.append(proc)

        w_par = np.zeros(n_features, dtype=np.float64)
        t0 = time.perf_counter()
        for it in range(n_gd_iters):
            # Send current weight vector to all persistent workers
            for conn in pipes:
                conn.send(w_par)
            # Receive partial gradients and synchronize
            grads = [conn.recv() for conn in pipes]
            g_total = sum(grads) / n_samples
            w_par -= learning_rate * g_total
        t_par_lr = time.perf_counter() - t0

        # Terminate workers
        for conn in pipes:
            conn.send(None)
        for proc in processes:
            proc.join()

        ms_per_iter = (t_par_lr / n_gd_iters) * 1e3
        sp = t_seq_lr / t_par_lr
        eff = sp / p_procs

        # Verify final loss equals sequential result within 1e-9
        z_par_final = X @ w_par
        p_par_final = 1.0 / (1.0 + np.exp(-np.clip(z_par_final, -30.0, 30.0)))
        par_loss = -float(np.mean(y * np.log(np.clip(p_par_final, 1e-15, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p_par_final, 1e-15, 1.0))))
        loss_diff = abs(par_loss - seq_loss)

        print(
            f"  p={p_procs:2d}: Time/Iter={ms_per_iter:.2f}ms [MEASURED] | Speedup={sp:.2f}x [MEASURED] | "
            f"Efficiency={eff:.2f} [MEASURED] | Loss Diff={loss_diff:.2e} (Tol < 1e-9)",
            flush=True,
        )

        if loss_diff > 1e-9:
            raise RuntimeError(f"Data parallelism loss mismatch at p={p_procs}: diff={loss_diff:.2e} > 1e-9")

        data_par_records.append({
            "procs": p_procs,
            "ms_per_iter": ms_per_iter,
            "speedup": sp,
            "efficiency": eff,
            "loss_diff": loss_diff,
        })
        records_csv.append({
            "experiment": "part_e_data_parallelism",
            "procs": p_procs,
            "speedup": sp,
            "efficiency": eff,
            "type": "measured",
        })

    # =======================================================================
    # Part (f): Pipeline Parallelism (3 Stages over 200 items)
    # =======================================================================
    print("\n" + "=" * 80)
    print("PART (F): Pipeline Parallelism (3 Stages: ~20ms, ~30ms, ~20ms over 200 items)")
    print("=" * 80, flush=True)

    n_pipe_items = 100 if args.quick else 200
    r_stage1 = calibrate_cpu_task(target_seconds=0.020, dim=150)
    r_stage2 = calibrate_cpu_task(target_seconds=0.030, dim=150)
    r_stage3 = calibrate_cpu_task(target_seconds=0.020, dim=150)

    # 1. Sequential execution
    print(f"  Running sequential pipeline over {n_pipe_items} items...", flush=True)
    t0 = time.perf_counter()
    seq_items = []
    for i in range(n_pipe_items):
        cpu_bound_task(r_stage1, dim=150)
        v1 = 0.0 + (1 * 1.5)
        cpu_bound_task(r_stage2, dim=150)
        v2 = v1 + (2 * 1.5)
        cpu_bound_task(r_stage3, dim=150)
        v3 = v2 + (3 * 1.5)
        seq_items.append((i, v3))
    t_seq_pipe = time.perf_counter() - t0
    thru_seq = n_pipe_items / t_seq_pipe

    # 2. Pipeline execution (3 processes connected by mp.Queue)
    print(f"  Running 3-stage concurrent process pipeline over {n_pipe_items} items...", flush=True)
    q_in = mp.Queue()
    q_12 = mp.Queue()
    q_23 = mp.Queue()
    q_out = mp.Queue()

    p1 = mp.Process(target=pipeline_stage_worker, args=(q_in, q_12, r_stage1, 1, 150))
    p2 = mp.Process(target=pipeline_stage_worker, args=(q_12, q_23, r_stage2, 2, 150))
    p3 = mp.Process(target=pipeline_stage_worker, args=(q_23, q_out, r_stage3, 3, 150))

    t0 = time.perf_counter()
    p1.start()
    p2.start()
    p3.start()

    # Feed input items
    for i in range(n_pipe_items):
        q_in.put((i, 0.0))
    q_in.put(None)  # sentinel to shutdown stage 1

    # Drain output
    pipe_items = []
    while len(pipe_items) < n_pipe_items:
        item = q_out.get()
        pipe_items.append(item)

    p1.join()
    p2.join()
    p3.join()
    t_par_pipe = time.perf_counter() - t0
    thru_par = n_pipe_items / t_par_pipe

    pipe_speedup = t_seq_pipe / t_par_pipe
    ideal_speedup = (0.020 + 0.030 + 0.020) / 0.030  # ~2.33x

    # Verify identical output items
    pipe_items_sorted = sorted(pipe_items, key=lambda x: x[0])
    for (idx_s, val_s), (idx_p, val_p) in zip(seq_items, pipe_items_sorted):
        assert idx_s == idx_p, f"Pipeline item mismatch: {idx_s} != {idx_p}"
        assert abs(val_s - val_p) < 1e-12, f"Pipeline value mismatch: {val_s} != {val_p}"

    print(f"  Sequential Pipeline: Time={t_seq_pipe:.3f}s [MEASURED], Throughput={thru_seq:.2f} items/s [MEASURED]")
    print(f"  Concurrent Pipeline: Time={t_par_pipe:.3f}s [MEASURED], Throughput={thru_par:.2f} items/s [MEASURED]")
    print(f"  Pipeline Speedup:    {pipe_speedup:.2f}x [MEASURED] (vs Ideal Bottleneck Limit = {ideal_speedup:.2f}x)")
    print(f"  Output Integrity:    All {n_pipe_items} items identical to sequential [PASS]", flush=True)

    records_csv.append({
        "experiment": "part_f_pipeline",
        "procs": 3,
        "speedup": pipe_speedup,
        "throughput_par": thru_par,
        "type": "measured",
    })

    # =======================================================================
    # Part (g): Model Parallelism (Column-Sharded y = tanh(X W) via Threads)
    # =======================================================================
    print("\n" + "=" * 80)
    print("PART (G): Model Parallelism (Column-Sharded Matrix Layer y = tanh(X W))")
    print("=" * 80, flush=True)

    rng_m = np.random.default_rng(42)
    B, Din, Dout = 2000, 2000, 4000
    X_m = rng_m.standard_normal((B, Din), dtype=np.float64)
    W_m = rng_m.standard_normal((Din, Dout), dtype=np.float64)

    # Sequential unsplit
    t0 = time.perf_counter()
    y_seq_m = np.tanh(X_m @ W_m)
    t_seq_model = time.perf_counter() - t0
    print(f"  Sequential (unsplit 1 thread): Time={t_seq_model*1e3:.1f}ms [MEASURED]")

    def thread_worker(X_in: np.ndarray, W_slice: np.ndarray, out_slice: np.ndarray):
        out_slice[:] = np.tanh(X_in @ W_slice)

    for p_threads in [1, 2, 4]:
        cols_per_worker = Dout // p_threads
        y_par_m = np.empty((B, Dout), dtype=np.float64)
        threads: List[threading.Thread] = []

        t0 = time.perf_counter()
        for i in range(p_threads):
            c_start = i * cols_per_worker
            c_end = (i + 1) * cols_per_worker
            t = threading.Thread(target=thread_worker, args=(X_m, W_m[:, c_start:c_end], y_par_m[:, c_start:c_end]))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        t_par_model = time.perf_counter() - t0

        sp_model = t_seq_model / t_par_model
        max_diff = float(np.max(np.abs(y_seq_m - y_par_m)))

        print(
            f"  p={p_threads:1d} threads: Time={t_par_model*1e3:6.1f}ms [MEASURED] | "
            f"Speedup={sp_model:.2f}x [MEASURED] | Max Abs Diff={max_diff:.2e} (Tol < 1e-10)",
            flush=True,
        )

        if max_diff > 1e-10:
            raise RuntimeError(f"Model parallelism numerical mismatch at p={p_threads}: max diff={max_diff:.2e} > 1e-10")

        records_csv.append({
            "experiment": "part_g_model_parallelism",
            "threads": p_threads,
            "speedup": sp_model,
            "type": "measured",
        })

    # =======================================================================
    # Save Results CSV and Plots
    # =======================================================================
    df_out = pd.DataFrame(records_csv)
    csv_path = RESULTS_DIR / "trackD_parallel.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved parallel benchmark records to {csv_path}", flush=True)

    # Plotting
    print("Generating parallel scaling figures...", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Strong & Scaled Speedup vs Amdahl / Gustafson Fit
    p_fine = np.linspace(1, max(procs_list), 100)
    amdahl_curve = [amdahl_speedup(s_amdahl, p) for p in p_fine]
    gustafson_curve = [gustafson_speedup(s_gustafson, p) for p in p_fine]

    ax1.plot(p_fine, p_fine, "--", color="gray", alpha=0.7, label="Ideal Linear ($S=p$)")
    ax1.plot(procs_arr, speedups_arr, "o", color="#1f77b4", markersize=7, label="Strong Scaling [MEASURED]")
    ax1.plot(p_fine, amdahl_curve, "-", color="#1f77b4", label=f"Amdahl Fit ($s={s_amdahl:.3f}$)")
    ax1.plot([r["procs"] for r in gustafson_results], [r["scaled_speedup"] for r in gustafson_results], "s", color="#2ca02c", markersize=7, label="Weak Scaling [MEASURED]")
    ax1.plot(p_fine, gustafson_curve, "-", color="#2ca02c", label=f"Gustafson Fit ($s={s_gustafson:.3f}$)")

    ax1.set_xlabel("Number of Processors ($p$)", fontsize=11)
    ax1.set_ylabel("Speedup ($S$)", fontsize=11)
    ax1.set_title("Parallel Speedup & Law Fitting", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    # Subplot 2: Makespan by Scheduling Strategy
    strategies = [r["strategy"] for r in lb_summary]
    makespans = [r["makespan_s"] for r in lb_summary]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    bars = ax2.bar(strategies, makespans, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_ylabel("Makespan (seconds) [MEASURED]", fontsize=11)
    ax2.set_title("Makespan by Load Balancing Strategy (p=4)", fontsize=12)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "trackD_parallel_scaling.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_path}", flush=True)

    print("\nExperiment 2 completed successfully.", flush=True)


if __name__ == "__main__":
    main()
