#!/usr/bin/env python3
"""
trackG_xv6/run_bench.py

Automated benchmark runner for xv6-riscv schedbench (Track G).
Tests both RR (Round-Robin) and LOTTERY schedulers under CPUS=1.
Parses RESULT lines into results/trackG_bench.csv and prints summary tables.

Must be run inside WSL Ubuntu from the repo root:
    python3 trackG_xv6/run_bench.py
    python3 trackG_xv6/run_bench.py --sched LOTTERY
    python3 trackG_xv6/run_bench.py --parse-only
"""

import os
import sys
import time
import math
import fcntl
import select
import signal
import argparse
import subprocess


def get_qemu_version():
    """Retrieve and return the installed QEMU version string."""
    try:
        res = subprocess.run(
            ["qemu-system-riscv64", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return res.stdout.strip().splitlines()[0]
    except Exception as e:
        return f"Unknown QEMU version ({e})"


def kill_leftover_qemu():
    """Kill any leftover qemu-system-riscv64 processes to prevent device/file lock contention."""
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "qemu-system-riscv64"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(0.5)
    except Exception:
        pass


def verify_no_qemu_running():
    """Verify that no lingering qemu-system-riscv64 process exists."""
    try:
        res = subprocess.run(
            ["pgrep", "-f", "qemu-system-riscv64"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        pids = res.stdout.strip().split()
        if pids:
            # Force kill if any remaining
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:
                    pass
            time.sleep(0.5)
    except Exception:
        pass


def kill_process_tree(proc, pgid=None):
    """Safely terminate the process and its entire process group."""
    target_pgid = pgid
    if target_pgid is None and proc is not None:
        try:
            target_pgid = os.getpgid(proc.pid)
        except Exception:
            pass

    if target_pgid is not None:
        try:
            os.killpg(target_pgid, signal.SIGTERM)
            time.sleep(0.5)
            os.killpg(target_pgid, signal.SIGKILL)
        except Exception:
            pass

    kill_leftover_qemu()


def build_xv6(sched_name, xv6_dir):
    """
    Run make clean and build kernel/kernel and fs.img explicitly.
    Verifies that the build step finishes successfully before QEMU is launched.
    """
    print(f"\n--- Cleaning and building xv6 for SCHED={sched_name} ---")
    clean_res = subprocess.run(
        ["make", "clean"],
        cwd=xv6_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if clean_res.returncode != 0:
        raise RuntimeError(f"make clean failed:\n{clean_res.stderr}")

    build_cmd = ["make", f"SCHED={sched_name}", "kernel/kernel", "fs.img"]
    print(f"Executing build: {' '.join(build_cmd)} in {xv6_dir}")
    build_res = subprocess.run(
        build_cmd,
        cwd=xv6_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if build_res.returncode != 0:
        raise RuntimeError(f"Build failed for SCHED={sched_name}:\n{build_res.stderr}")
    print("Build succeeded cleanly.")


def run_xv6_benchmark(sched_name, xv6_dir, results_dir, timeout_sec=1200):
    """
    Build and execute schedbench in xv6 under CPUS=1 for the given scheduler.
    Uses non-blocking raw os.read with select to avoid stdio buffer desync.
    Returns (raw_output_text, result_lines).
    """
    print(f"\n=======================================================")
    print(f"Starting QEMU benchmark for SCHED={sched_name}")
    print(f"=======================================================")

    # Ensure no leftover QEMU processes are running
    kill_leftover_qemu()
    verify_no_qemu_running()

    # Step 1: Clean build and verify
    build_xv6(sched_name, xv6_dir)

    # Step 2: Launch QEMU
    qemu_cmd = ["make", "qemu", f"SCHED={sched_name}", "CPUS=1"]
    print(f"Executing QEMU: {' '.join(qemu_cmd)}")

    proc = subprocess.Popen(
        qemu_cmd,
        cwd=xv6_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,  # Raw binary mode for direct os.read
        bufsize=0,
        preexec_fn=os.setsid
    )
    pgid = os.getpgid(proc.pid)

    # Set stdout to non-blocking mode
    stdout_fd = proc.stdout.fileno()
    flags = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
    fcntl.fcntl(stdout_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    full_output_buf = []
    current_buf = ""
    prompt_found = False
    start_time = time.time()

    try:
        # Step A: Wait for shell prompt (searching for "\n$ " or ending in "$ ")
        print("Waiting for xv6 shell prompt (120 s timeout)...")
        while time.time() - start_time < 120:
            if proc.poll() is not None:
                raise RuntimeError(f"QEMU exited prematurely with code {proc.returncode}")

            r, _, _ = select.select([stdout_fd], [], [], 0.5)
            if r:
                try:
                    chunk = os.read(stdout_fd, 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    break
                text_chunk = chunk.decode("utf-8", errors="replace")
                full_output_buf.append(text_chunk)
                current_buf += text_chunk

                if "\n$ " in current_buf or current_buf.endswith("$ "):
                    prompt_found = True
                    break

        if not prompt_found:
            full_out = "".join(full_output_buf)
            timeout_file = os.path.join(results_dir, f"trackG_timeout_{sched_name}.txt")
            with open(timeout_file, "w", encoding="utf-8") as f:
                f.write(full_out)
            lines = full_out.splitlines()
            last_30 = "\n".join(lines[-30:]) if len(lines) >= 30 else full_out
            raise TimeoutError(
                f"Timed out waiting for xv6 shell prompt '$ ' after 120 s for SCHED={sched_name}.\n"
                f"Full output saved to {timeout_file}.\n"
                f"--- Last 30 lines of output ---\n{last_30}\n------------------------------"
            )

        print("Shell prompt detected. Sending 'schedbench' command...")
        proc.stdin.write(b"schedbench\n")
        proc.stdin.flush()

        # Step B: Read output until BENCH_DONE
        bench_done = False
        bench_start = time.time()
        print("Running schedbench workloads (timeout 20 min)...")

        while time.time() - bench_start < timeout_sec:
            if proc.poll() is not None:
                break

            r, _, _ = select.select([stdout_fd], [], [], 0.5)
            if r:
                try:
                    chunk = os.read(stdout_fd, 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    break
                text_chunk = chunk.decode("utf-8", errors="replace")
                full_output_buf.append(text_chunk)
                current_buf += text_chunk

                if "BENCH_DONE" in current_buf:
                    bench_done = True
                    # Drain any final residual bytes
                    time.sleep(0.5)
                    while True:
                        r2, _, _ = select.select([stdout_fd], [], [], 0.1)
                        if not r2:
                            break
                        try:
                            rem = os.read(stdout_fd, 1024)
                            if not rem:
                                break
                            full_output_buf.append(rem.decode("utf-8", errors="replace"))
                        except Exception:
                            break
                    break

        if not bench_done:
            print(f"WARNING: BENCH_DONE not detected within {timeout_sec}s timeout.")

        # Step C: Send Ctrl-a x (b"\x01x") to quit QEMU
        print("Sending QEMU exit sequence (Ctrl-a x)...")
        try:
            proc.stdin.write(b"\x01x")
            proc.stdin.flush()
        except Exception:
            pass

        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("QEMU did not exit in 10s, killing process group...")
            kill_process_tree(proc, pgid)

    finally:
        kill_process_tree(proc, pgid)
        # Final cleanup
        try:
            subprocess.run(["make", "clean"], cwd=xv6_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    full_output_text = "".join(full_output_buf)
    result_lines = [
        line.strip()
        for line in full_output_text.splitlines()
        if line.strip().startswith("RESULT,")
    ]
    return full_output_text, result_lines


def mean_std(values):
    """Return (mean, sample_std) for a list of numbers."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    if n <= 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(var)


def parse_and_summarize(all_results):
    """
    Parse result lines into structured CSV records and print summary tables.
    all_results: dict mapping sched_name -> list of RESULT lines
    """
    csv_rows = [["sched", "workload", "rep", "metric", "value"]]

    data_w1 = {}
    data_w2 = {}
    data_w3 = {}

    for sched, lines in all_results.items():
        data_w1[sched] = {}
        data_w2[sched] = {}
        data_w3[sched] = {}

        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue

            wl = parts[1]
            rep = int(parts[2])

            if wl == "calib":
                # RESULT,calib,0,iters_per_tick,<val>
                metric = parts[3]
                val = parts[4]
                csv_rows.append([sched, wl, str(rep), metric, val])

            elif wl == "w1":
                # RESULT,w1,<rep>,work_t30,<n>,work_t20,<n>,work_t10,<n>
                w30 = int(parts[4])
                w20 = int(parts[6])
                w10 = int(parts[8])
                data_w1[sched][rep] = {30: w30, 20: w20, 10: w10}
                csv_rows.append([sched, wl, str(rep), "work_t30", str(w30)])
                csv_rows.append([sched, wl, str(rep), "work_t20", str(w20)])
                csv_rows.append([sched, wl, str(rep), "work_t10", str(w10)])

            elif wl == "w2":
                # RESULT,w2,<rep>,variant,<a|b>,turnaround_ticks,<n>
                variant = parts[4]
                ticks = int(parts[6])
                if rep not in data_w2[sched]:
                    data_w2[sched][rep] = {}
                data_w2[sched][rep][variant] = ticks
                csv_rows.append([sched, wl, str(rep), f"turnaround_{variant}", str(ticks)])

            elif wl == "w3":
                # RESULT,w3,<rep>,works,<n1>,<n2>,...,<n8>
                works = [int(x) for x in parts[4:]]
                data_w3[sched][rep] = works
                for i, w in enumerate(works):
                    csv_rows.append([sched, wl, str(rep), f"work_{i}", str(w)])

    # Print summary tables
    print("\n" + "=" * 76)
    print("BENCHMARK SUMMARY TABLES")
    print("=" * 76)

    # -------------------------------------------------------------
    # W1: Proportional Share Summary
    # -------------------------------------------------------------
    print("\n### Workload 1: Proportional Share (Tickets: 30, 20, 10 over 60 ticks)")
    print(f"{'Scheduler':<10} {'Ticket Class':<14} {'Expected Share':<18} {'Observed Share (Mean ± Std)':<30}")
    print("-" * 76)

    for sched in ["RR", "LOTTERY"]:
        if sched not in data_w1 or not data_w1[sched]:
            continue

        shares = {30: [], 20: [], 10: []}
        maes = []

        for rep in sorted(data_w1[sched].keys()):
            wdict = data_w1[sched][rep]
            tot = sum(wdict.values())
            if tot == 0:
                continue
            s30 = wdict[30] / tot
            s20 = wdict[20] / tot
            s10 = wdict[10] / tot
            shares[30].append(s30)
            shares[20].append(s20)
            shares[10].append(s10)

            if sched == "LOTTERY":
                exp30, exp20, exp10 = 30 / 60, 20 / 60, 10 / 60
            else:
                exp30, exp20, exp10 = 1 / 3, 1 / 3, 1 / 3

            mae = (abs(s30 - exp30) + abs(s20 - exp20) + abs(s10 - exp10)) / 3.0
            maes.append(mae)

        for tc in [30, 20, 10]:
            if sched == "LOTTERY":
                exp_str = f"{tc}/60 ({tc/60:.3f})"
            else:
                exp_str = "1/3 (0.333)"
            m, s = mean_std(shares[tc])
            print(f"{sched:<10} {tc:<14} {exp_str:<18} {m:.4f} ± {s:.4f}")

        m_mae, s_mae = mean_std(maes)
        print(f"  -> {sched} Mean Absolute Error (MAE) vs expected share: {m_mae:.4f} ± {s_mae:.4f}")
        print("-" * 76)

    # -------------------------------------------------------------
    # W2: Turnaround Time Summary
    # -------------------------------------------------------------
    print("\n### Workload 2: Turnaround Time (Short job of 15 calibrated ticks against 4 hogs)")
    print(f"{'Scheduler':<10} {'Variant':<10} {'Tickets':<10} {'Turnaround Ticks (Mean ± Std)':<30}")
    print("-" * 64)

    for sched in ["RR", "LOTTERY"]:
        if sched not in data_w2 or not data_w2[sched]:
            continue

        reps = sorted(data_w2[sched].keys())
        ta_list = [data_w2[sched][r]['a'] for r in reps if 'a' in data_w2[sched][r]]
        tb_list = [data_w2[sched][r]['b'] for r in reps if 'b' in data_w2[sched][r]]

        ma, sa = mean_std(ta_list)
        mb, sb = mean_std(tb_list)

        print(f"{sched:<10} {'a':<10} {'10':<10} {ma:.2f} ± {sa:.2f} ticks")
        print(f"{sched:<10} {'b':<10} {'40':<10} {mb:.2f} ± {sb:.2f} ticks")
        speedup = (ma / mb) if mb > 0 else 0.0
        print(f"  -> Boost speedup (variant a / variant b): {speedup:.2f}x")
        print("-" * 64)

    # -------------------------------------------------------------
    # W3: Fairness Summary
    # -------------------------------------------------------------
    print("\n### Workload 3: Short-Term Fairness (8 equal children with 10 tickets over 60 ticks)")
    print(f"{'Scheduler':<10} {'Jain Index (Mean ± Std)':<26} {'CV (Std/Mean) (Mean ± Std)':<28}")
    print("-" * 68)

    for sched in ["RR", "LOTTERY"]:
        if sched not in data_w3 or not data_w3[sched]:
            continue

        jains = []
        cvs = []

        for rep in sorted(data_w3[sched].keys()):
            works = data_w3[sched][rep]
            if not works or len(works) < 8:
                continue
            n = len(works)
            sum_x = sum(works)
            sum_sq = sum(w * w for w in works)
            j = (sum_x * sum_x) / (n * sum_sq) if sum_sq > 0 else 0.0
            jains.append(j)

            # Coefficient of Variation (CV = std / mean)
            m_w = sum_x / n
            var_w = sum((w - m_w) ** 2 for w in works) / (n - 1) if n > 1 else 0.0
            cv = (math.sqrt(var_w) / m_w) if m_w > 0 else 0.0
            cvs.append(cv)

        mj, sj = mean_std(jains)
        mc, sc = mean_std(cvs)
        print(f"{sched:<10} {mj:.4f} ± {sj:.4f}             {mc:.4f} ± {sc:.4f}")
    print("-" * 68)

    return csv_rows


def load_raw_file(raw_path):
    """Read a raw output file and extract RESULT lines."""
    if not os.path.exists(raw_path):
        return []
    res_lines = []
    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("RESULT,"):
                res_lines.append(stripped)
    return res_lines


def main():
    parser = argparse.ArgumentParser(description="Track G: xv6 Lottery Scheduler Benchmark Suite")
    parser.add_argument(
        "--sched",
        choices=["RR", "LOTTERY"],
        default=None,
        help="Run benchmark only for the specified scheduler (default: both)"
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Skip running QEMU; re-parse existing raw files into CSV and summary tables"
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    xv6_dir = os.path.join(repo_root, "trackG_xv6", "xv6-riscv")
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=======================================================")
    print("Track G: xv6 Lottery Scheduler Benchmark Suite")
    print(f"Repo root:    {repo_root}")
    print(f"xv6 dir:      {xv6_dir}")
    print(f"Results dir:  {results_dir}")
    print(f"QEMU Version: {get_qemu_version()}")
    print("=======================================================")

    all_results = {}
    schedulers_to_run = [args.sched] if args.sched else ["RR", "LOTTERY"]

    if args.parse_only:
        print("\n[--parse-only mode: Skipping QEMU execution]")
        for sched in ["RR", "LOTTERY"]:
            raw_path = os.path.join(results_dir, f"trackG_raw_{sched}.txt")
            if os.path.exists(raw_path):
                print(f"Loading raw results from {raw_path}")
                all_results[sched] = load_raw_file(raw_path)
            else:
                print(f"Note: {raw_path} does not exist, skipping.")
    else:
        for sched in schedulers_to_run:
            raw_out, res_lines = run_xv6_benchmark(sched, xv6_dir, results_dir)
            all_results[sched] = res_lines

            # Save raw output
            raw_path = os.path.join(results_dir, f"trackG_raw_{sched}.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw_out)
            print(f"Saved raw console output to {raw_path}")

        # Also load the other scheduler's raw results if already present
        for sched in ["RR", "LOTTERY"]:
            if sched not in all_results:
                raw_path = os.path.join(results_dir, f"trackG_raw_{sched}.txt")
                if os.path.exists(raw_path):
                    all_results[sched] = load_raw_file(raw_path)

    # Parse and write results/trackG_bench.csv
    csv_rows = parse_and_summarize(all_results)
    csv_path = os.path.join(results_dir, "trackG_bench.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(",".join(row) + "\n")
    print(f"\nAll benchmark records successfully written to {csv_path}")


if __name__ == "__main__":
    main()
