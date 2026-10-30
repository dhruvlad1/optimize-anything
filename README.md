# Optimize Anything

Student: Dhruv Lad | Roll No: 241070037 | Batch: B  
Subject: Optimization Techniques | Lab Experiment 8: Optimize Anything (14-Day Vibe-Coding Challenge)

Optimization Techniques, Assignment 8: a 14-day vibe-coding challenge compressed into 4 days, systematically exploring diverse mathematical, algorithmic, and systems optimization domains across continuous optimization, discrete algorithms, quantum computing, parallel processing, urban routing, compiler optimization passes, operating systems internals, database query engines, and software engineering workflows.

## Setup & How to Run

Execute the following setup commands in Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tools/check_env.py
```

*Note:* If PowerShell blocks script activation, allow it by running:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Track A: Continuous Optimization

> **Run exp3_sensitivity first** — it writes the tuned learning rates consumed by the other experiments.

```powershell
# 1. Sensitivity sweep & tuned-LR generation (run first)
python -m trackA_continuous.exp3_sensitivity

# 2. Convex benchmark (reads tuned LRs)
python -m trackA_continuous.exp1_convex

# 3. Non-convex experiments
python -m trackA_continuous.exp2_nonconvex

# 4. Noisy gradient & finite-sum experiments
python -m trackA_continuous.exp4_noisy

# 5. Machine learning optimization benchmark on make_moons
python -m trackA_continuous.exp5_backprop_ml

# 6. Curvature methods (HVP, Gauss-Newton, Empirical Fisher)
python -m trackA_continuous.exp6_curvature

# 7. Test suite (including test_optimizers, test_objectives, test_curvature)
pytest trackA_continuous/tests -v
pytest trackA_continuous/compare/test_backprop_compare.py -v
```

Results (CSV + PNG) are written to `results/`.

## Track B: Discrete Optimization & QUBO

```powershell
# 1. 0/1 Knapsack benchmarks (equivalence on n=15,20; scaling on n=50,200,1000)
python -m trackB_discrete.exp1_knapsack

# 2. QUBO, Max-Cut, and Ising benchmarks
python -m trackB_discrete.exp2_qubo

# 3. Test suite
pytest trackB_discrete/tests -v
```

Results (CSV + PNG + JSON instances) are written to `results/`.

## Track E: DabbaDash (Mumbai Last-Mile CVRP)

To execute baseline heuristics (driver order and nearest neighbour) across problem sizes $N \in [10, 20, 50, 100, 200]$:

```powershell
python -m trackE_mumbai_routing.run_baseline
# Writes: results/trackE_baseline.csv, results/trackE_baseline_map.png
```

To run the comprehensive comparison benchmark (exact ILP on small instances, scaling study on $N \in [20, 50, 100, 200]$, multi-core parallelism, and route maps):

```powershell
python -m trackE_mumbai_routing.run_comparison
# Writes: results/trackE_comparison.csv, results/trackE_scaling.png, results/trackE_routes_n50.png
```

To run the Track E test suite:

```powershell
python -m pytest trackE_mumbai_routing/ -v
```

## Track F: MiniLang Compiler Optimization

To execute baseline compilation, run on the AST oracle and VM, and generate benchmark performance records:

```powershell
python trackF_compiler/run_baseline.py
```

To compile a MiniLang program with optimization passes:

```powershell
# Full fixed-order pipeline (constant_fold -> cse -> licm -> strength_reduce -> dead_code_elim)
python -m trackF_compiler.minilang.compiler <file.ml> --opt 1

# Custom pass sequence
python -m trackF_compiler.minilang.compiler <file.ml> --passes constant_fold,cse,dead_code_elim
```

To run the differential fuzzer (300 randomly generated programs verified against the AST oracle):

```powershell
python -m trackF_compiler.fuzz
```

To search for the best pass ordering (exhaustive permutations + greedy; uses scaled-down benchmarks for speed):

```powershell
python -m trackF_compiler.exp_pass_order
# Writes: results/trackF_pass_order.csv
```

To run the full ablation benchmark (baseline vs single passes vs fixed order vs best order):

```powershell
python -m trackF_compiler.run_optimized
# Writes: results/trackF_optimized.csv  results/trackF_optimized_bar.png
```

To run the full test suite for Track F (18 tests):

```powershell
python -m pytest trackF_compiler/ -v
```

## Track G: Operating Systems Optimization (xv6 Lottery Scheduler)

To build and run the comprehensive scheduling benchmark suite comparing Round-Robin and Lottery scheduling in xv6-riscv under single-core QEMU (`CPUS=1`):

```bash
# Inside WSL Ubuntu (from repository root):
python3 trackG_xv6/run_bench.py
```

Outputs written to:
- `results/trackG_raw_RR.txt` — raw QEMU console log for default Round-Robin scheduler
- `results/trackG_raw_LOTTERY.txt` — raw QEMU console log for Lottery scheduler
- `results/trackG_bench.csv` — parsed measurement records across calibration, W1, W2, and W3

## Track H: DBMS Optimization (Index Selection)

Requires PostgreSQL running in WSL2 with database `optdb`, user `optuser`.

```powershell
# Set password (never commit this)
$env:PGPASSWORD = "your_password"

# (WSL2) start PostgreSQL
wsl sudo service postgresql start

# Generate synthetic data and load into DB
python -m trackH_dbms.load_data

# Run baseline workload (no secondary indexes)
python -m trackH_dbms.run_baseline

# Build offline cost model across candidate secondary indexes
python -m trackH_dbms.build_cost_model
# Writes: results/trackH_cost_model.json, results/trackH_candidate_benefits.csv

# Run index selection optimization across budgets (greedy, DP, ILP) and measure physical performance
python -m trackH_dbms.run_index_selection
# Writes: results/trackH_index_selection.csv, results/trackH_plans_after/, results/trackH_latency_vs_budget.png

# Run offline index selection solver test suite (no database required)
python -m pytest trackH_dbms/tests/test_index_solvers.py -v

# Run full database test suite (skipped gracefully if DB unavailable)
python -m pytest trackH_dbms/tests -v
```

Results are written to:
- `results/trackH_baseline.csv` — per-query latency summary
- `results/trackH_baseline_plans/` — EXPLAIN ANALYZE JSON plans
- `results/trackH_cost_model.json` — per-query per-candidate offline cost matrix
- `results/trackH_candidate_benefits.csv` — candidate sizes, build times, and benefits per MB
- `results/trackH_index_selection.csv` — physical performance across budgets and methods
- `results/trackH_plans_after/` — post-optimization EXPLAIN plans for top improved queries
- `results/trackH_latency_vs_budget.png` — latency vs storage budget trade-off curves
- `results/trackH_environment.json` — PostgreSQL version and settings

## Commit Convention

All repository commits follow a standardized prefix convention to clearly distinguish AI-assisted generation from manual work:

- `[AI:<tool>] <description>`: For changes generated by an AI tool (e.g., `[AI:antigravity] Setup project repository structure`, `[AI:claude] Implement simulated annealing for Track B`).
- `[MANUAL] <description>`: For human edits, independent verifications, manual bug fixes, or handwritten notes.

## AI Tools Register

| Tool | Model/version | Tracks | Dates | Kind of work (planning/code generation/debugging/testing/documentation) |
| --- | --- | --- | --- | --- |
| Claude | Claude Sonnet 5.5 | 00_setup, E, F, G, H | 2026-10-27 to 2026-10-28 | Planning, prompt writing, review of results, debugging, documentation (prompt logs, failure log) |
| Antigravity | Gemini 4.8 Flash | 00_setup, E, F, H | 2026-10-27 to 2026-10-28 | Code generation, running tests and benchmarks |
