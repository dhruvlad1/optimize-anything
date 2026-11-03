"""trackD_parallel/parallel_ops.py
--------------------------------
Shared routines, mathematical laws, and picklable worker entry-points for Track D.
All worker functions are defined at the module level so they can be safely spawned
by Python's multiprocessing engine on Windows.
"""

from __future__ import annotations

import collections
import os
import queue
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit, minimize


# ===========================================================================
# 1. Parallel Laws and Metrics
# ===========================================================================

def amdahl_speedup(s: float, p: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """Amdahl's law for fixed workload:
    S(p) = 1 / (s + (1 - s) / p).
    """
    p_arr = np.asarray(p, dtype=float)
    denom = s + (1.0 - s) / p_arr
    res = 1.0 / denom
    if np.ndim(p) == 0:
        return float(res)
    return res


def karp_flatt_serial_fraction(p: int, speedup: float) -> float:
    """Karp-Flatt experimentally determined serial fraction:
    e = (1 / S - 1 / p) / (1 - 1 / p).
    Defined for p > 1.
    """
    if p <= 1:
        raise ValueError(f"Karp-Flatt is only defined for p > 1, got p={p}")
    numerator = (1.0 / speedup) - (1.0 / p)
    denominator = 1.0 - (1.0 / p)
    return float(numerator / denominator)


def gustafson_speedup(s: float, p: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """Gustafson's law for scaled workload:
    S_scaled(p) = p - s * (p - 1).
    """
    p_arr = np.asarray(p, dtype=float)
    res = p_arr - s * (p_arr - 1.0)
    if np.ndim(p) == 0:
        return float(res)
    return res


def fit_amdahl(procs: Sequence[int], speedups: Sequence[float]) -> float:
    """Fit the serial fraction s in Amdahl's law via non-linear least squares."""
    p_arr = np.asarray(procs, dtype=float)
    s_arr = np.asarray(speedups, dtype=float)

    def obj(s_val: float) -> float:
        pred = amdahl_speedup(s_val, p_arr)
        return float(np.sum((pred - s_arr) ** 2))

    res = minimize(obj, x0=0.05, bounds=[(0.0, 1.0)], method="L-BFGS-B")
    return float(res.x[0])


def fit_gustafson(procs: Sequence[int], speedups: Sequence[float]) -> float:
    """Fit the serial fraction s in Gustafson's law via linear least squares."""
    p_arr = np.asarray(procs, dtype=float)
    s_arr = np.asarray(speedups, dtype=float)

    # S(p) = p - s * (p - 1) => s * (p - 1) = p - S(p)
    # y = p - S(p), x = p - 1
    mask = p_arr > 1
    if not np.any(mask):
        return 0.0
    x = p_arr[mask] - 1.0
    y = p_arr[mask] - s_arr[mask]
    s_fitted = float(np.sum(x * y) / np.sum(x ** 2))
    return float(np.clip(s_fitted, 0.0, 1.0))


# ===========================================================================
# 2. CPU-Bound Task Kernels and Calibration
# ===========================================================================

def cpu_bound_task(reps: int, dim: int = 250) -> float:
    """Pure NumPy CPU-bound task performing reps of (dim x dim) matrix products.
    Returns elapsed wall-clock seconds.
    """
    t0 = time.perf_counter()
    A = np.ones((dim, dim), dtype=np.float64) * 1.0001
    B = np.ones((dim, dim), dtype=np.float64) * 0.9999
    acc = 0.0
    for _ in range(reps):
        C = A @ B
        acc += C[0, 0]
        A[0, 0] = C[0, 0] * 1e-4 + 1.0
    elapsed = time.perf_counter() - t0
    return elapsed


def run_cpu_task_wrapper(args: Union[int, Tuple[int, int]]) -> float:
    """Top-level picklable entry point for multiprocessing pool mapping.
    args can be reps (int, default dim=250) or (reps, dim) tuple.
    """
    if isinstance(args, tuple):
        reps, dim = args
    else:
        reps, dim = args, 250
    return cpu_bound_task(reps=reps, dim=dim)


def calibrate_cpu_task(target_seconds: float = 0.5, dim: int = 250) -> int:
    """Dynamically calibrate the number of matrix product loops to reach target_seconds."""
    # Warm-up / baseline 20 reps
    t0 = time.perf_counter()
    _ = cpu_bound_task(20, dim=dim)
    elapsed = max(1e-6, time.perf_counter() - t0)
    rate = 20.0 / elapsed  # reps per second

    target_reps = max(1, int(target_seconds * rate))
    return target_reps


def load_balance_task(args: Tuple[int, int, int]) -> Tuple[int, int, float]:
    """Task for load-balancing benchmarks.
    args: (task_id, reps, dim)
    Returns: (task_id, pid, elapsed_seconds)
    """
    task_id, reps, dim = args
    pid = os.getpid()
    elapsed = cpu_bound_task(reps, dim=dim)
    return task_id, pid, elapsed


def batch_load_balance_task(task_list: List[Tuple[int, int, int]]) -> Tuple[int, float]:
    """Execute a list of tasks sequentially in one worker process.
    Returns: (pid, total_elapsed_seconds)
    """
    pid = os.getpid()
    t0 = time.perf_counter()
    for task_id, reps, dim in task_list:
        cpu_bound_task(reps, dim=dim)
    total_elapsed = time.perf_counter() - t0
    return pid, total_elapsed


# ===========================================================================
# 3. Scheduling and Work-Stealing Simulation
# ===========================================================================

def lpt_schedule(durations: Sequence[float], p: int) -> List[List[int]]:
    """Longest Processing Time (LPT) greedy partition of task indices into p workers.
    Sort tasks by duration descending and assign each to the worker with minimum total load.
    """
    sorted_indices = sorted(range(len(durations)), key=lambda idx: durations[idx], reverse=True)
    worker_tasks: List[List[int]] = [[] for _ in range(p)]
    worker_loads = [0.0] * p

    for idx in sorted_indices:
        min_worker = int(np.argmin(worker_loads))
        worker_tasks[min_worker].append(idx)
        worker_loads[min_worker] += durations[idx]

    return worker_tasks


def simulate_static_schedule(durations: Sequence[float], p: int) -> float:
    """Simulated makespan for static contiguous split of tasks across p workers."""
    N = len(durations)
    base, rem = divmod(N, p)
    worker_times: List[float] = []
    start = 0
    for w in range(p):
        count = base + (1 if w < rem else 0)
        chunk = durations[start : start + count]
        worker_times.append(float(sum(chunk)))
        start += count
    return float(max(worker_times))


def simulate_dynamic_schedule(durations: Sequence[float], p: int) -> float:
    """Simulated makespan for ideal centralized dynamic queue (greedy next-available worker)."""
    worker_times = [0.0] * p
    for d in durations:
        w_min = int(np.argmin(worker_times))
        worker_times[w_min] += d
    return float(max(worker_times))


def simulate_work_stealing(durations: Sequence[float], p: int) -> Dict[str, Any]:
    """Discrete-event simulation of per-worker double-ended queues (deques).
    Tasks are initially partitioned equally among workers.
    When a worker's deque becomes empty, it steals a task from the tail of the longest deque.

    Returns:
        {
            "simulated_makespan": float,
            "num_steals": int,
            "worker_times": List[float],
            "total_work_executed": float,
            "theoretical_lower_bound": float,
        }
    """
    dur_list = list(durations)
    N = len(dur_list)
    total_work = float(sum(dur_list))
    max_single_task = float(max(dur_list)) if dur_list else 0.0
    lower_bound = max(total_work / p, max_single_task)

    # Initial equal partition into worker deques
    deques: List[collections.deque] = [collections.deque() for _ in range(p)]
    for idx, d in enumerate(dur_list):
        deques[idx % p].append(d)

    worker_clocks = [0.0] * p
    steals = 0
    executed_work = 0.0

    # Simulation loop: advance until all deques are empty
    while any(len(dq) > 0 for dq in deques):
        # Pick the worker with the lowest clock that still has work or can steal
        candidates = [w for w in range(p) if len(deques[w]) > 0]
        if candidates:
            # Workers with tasks proceed
            w = min(candidates, key=lambda i: worker_clocks[i])
            task = deques[w].popleft()
            worker_clocks[w] += task
            executed_work += task
        else:
            break

        # Check idle workers and perform stealing
        for idle_w in range(p):
            if len(deques[idle_w]) == 0:
                # Find the worker with the longest deque
                longest_w = max(range(p), key=lambda i: len(deques[i]))
                if len(deques[longest_w]) > 1:
                    stolen_task = deques[longest_w].pop()  # steal from tail
                    deques[idle_w].append(stolen_task)
                    steals += 1
                    # Worker idle clock synchronizes to current simulation front if behind
                    if worker_clocks[idle_w] < worker_clocks[longest_w]:
                        worker_clocks[idle_w] = worker_clocks[longest_w]

    # Drain any remaining tasks
    for w in range(p):
        while deques[w]:
            task = deques[w].popleft()
            worker_clocks[w] += task
            executed_work += task

    makespan = float(max(worker_clocks))

    return {
        "simulated_makespan": makespan,
        "num_steals": steals,
        "worker_times": worker_clocks,
        "total_work_executed": executed_work,
        "theoretical_lower_bound": lower_bound,
    }


# ===========================================================================
# 4. Data Parallelism Persistent Worker
# ===========================================================================

def data_parallel_worker(pipe_conn: Any, X_shard: np.ndarray, y_shard: np.ndarray) -> None:
    """Persistent worker for data-parallel logistic regression.
    Keeps local shard in memory and communicates only weights and gradients via pipe.
    """
    while True:
        msg = pipe_conn.recv()
        if msg is None:
            break
        w = msg
        z = X_shard @ w
        # Numerically stable sigmoid
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        grad_partial = X_shard.T @ (p - y_shard)
        pipe_conn.send(grad_partial)


# ===========================================================================
# 5. Pipeline Parallelism Workers
# ===========================================================================

def pipeline_stage_worker(
    in_queue: Any,
    out_queue: Any,
    reps: int,
    stage_id: int,
    dim: int = 150,
) -> None:
    """Generic worker process for a stage in pipeline parallelism.
    Processes items from in_queue until encountering None sentinel.
    """
    while True:
        item = in_queue.get()
        if item is None:
            if out_queue is not None:
                out_queue.put(None)
            break
        item_id, val = item
        # Perform calibrated CPU work
        cpu_bound_task(reps, dim=dim)
        new_val = val + (stage_id * 1.5)
        if out_queue is not None:
            out_queue.put((item_id, new_val))
