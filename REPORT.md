# Optimize Anything - Technical Report

**Student:** Dhruv Lad | **Roll No:** 241070037 | **Batch:** B
**Subject:** Optimization Techniques | **Lab Experiment 8:** Optimize Anything (14-Day Vibe-Coding Challenge, completed in 4 days)
**Repository:** https://github.com/dhruvlad1/optimize-anything
**Demo video:** [TODO: paste link]

## 1. Overview

This report covers the ten compulsory tracks (A to J). Every track follows the same loop: measure a baseline, formulate the problem (variables, objective, constraints), choose a method, implement it with AI-assisted coding, verify it independently, compare against the baseline, and state the limits.

Every result in this report carries one of three labels:

- **MEASURED**: a timing, count or accuracy taken on this machine (Windows laptop, 12 logical cores, 10 physical cores; PostgreSQL and xv6 ran inside WSL2).
- **SIMULATED**: a quantum computation run on a classical simulator or noise model. No real quantum hardware was used anywhere in this work.
- **THEORETICAL**: a formula or bound that was derived, not measured.

Reproducibility: every experiment has fixed seeds and a command in README.md. Tests per track: A 58, B 7, C 16, D 16, E 14, F 18, H 49 (44 solver tests plus 5 database tests), I 14, J 5; Track G is verified by xv6 booting and passing usertests. Prompt logs are in `prompts/<track>/`, the failure log (23 entries) is in `failure_log.md`, and the AI tools register is in README.md.

Main limits that apply to the whole report: synthetic data in Tracks E, H and I; a single machine; 3 to 5 repetitions per stochastic experiment; the Chapter 6 checklist follows the topic list in the assignment, not the class slides (see CHECKLIST.md).

## 2. Track A - Continuous optimization

**Objective.** Compare first-order and second-order optimizers on convex and non-convex objectives, and show backpropagation and curvature methods on a small model. **Baseline:** plain gradient descent (GD) with step 1/L. Code: `trackA_continuous/`; results: `results/trackA_*`.

**Methods.** GD, SGD, mini-batch SGD, Momentum, Nesterov (NAG), AdaGrad, RMSProp, Adam, pure and damped Newton; backpropagation for a 2-layer network (two independent implementations, see Section 12); Hessian-vector products by finite differences, power iteration, Gauss-Newton, Levenberg-Marquardt (LM), empirical Fisher (full, diagonal, low-rank).

**Results (MEASURED).**

- GD with step 1/L on a quadratic contracts the objective gap by (1 - 1/kappa)^2 per step: measured 0.8073 against 0.8100 (kappa = 10) and 0.9801 against 0.9801 (kappa = 100). The first version of this check used the wrong theoretical target (1 - 1/kappa); this was my prompt's error and is failure log row 6.
- Saddle point: GD started at distance y0 from the saddle escapes in 31, 56, 104 and 152 iterations for y0 = 1e-1, 1e-2, 1e-4, 1e-6 (escape time grows with log(1/y0)). Pure Newton converges to the saddle; damped Newton escapes in 6 iterations.
- Himmelblau, 20 random starts: Adam reached the four minima 6/7/5/2 times, GD 3/7/5/5. Rosenbrock global minimum was reached by 5% (GD), 50% (Momentum) and 65% (Adam) of 20 starts; no first-order method reached the global minimum of Rastrigin.
- Noisy quadratic: the SGD noise floor scales with sigma squared (5.2e-6, 5.2e-4, 5.2e-2 for sigma 0.01, 0.1, 1). Finite-sum least squares: f* = 4.81e-5 (gradient norm 6e-15 at the lstsq solution); at equal sample-gradient budgets full-batch GD reaches about 1e-19 by 50n, while SGD and mini-batch SGD stay at noise floors of 1e-6 to 1e-7.
- Hessian-vector products match the explicit Hessian to 2.9e-11; power iteration finds the top eigenvalue (0.709578, relative error 3e-11); the most negative eigenvalue is -0.43 (non-convex).
- Non-linear least squares y = a exp(b x): Newton 5 iterations, Gauss-Newton 7, LM 7, GD 405, all reaching cost 3.0432e-2, equal to SciPy's optimum. From a poor start, plain Gauss-Newton stalls at cost 13.86 while LM converges (14 iterations).
- make_moons with the hand-written backpropagation: GD and Adam both reach 96% test accuracy; L2 = 0.01 gives 93%, L2 = 0.1 gives 81% (over-regularised).

**Correctness checks.** Finite-difference checks of every gradient and Hessian; hand-computed one-step updates; SciPy reference for Rosenbrock and least squares.

**Limits.** On Rosenbrock the learning-rate sweep ranks progress within 2000 iterations, not convergence (median gap about 4 for every method, 0 of 5 converged). The learning-rate grid is 1e-4 to 10. The comparison of Fisher preconditioners is inconclusive: tuned rates sat at grid edges, the test set has 100 samples, one seed (failure log row 9). Adam's noisy-quadratic numbers changed between runs without a verified cause.

## 3. Track B - Discrete optimization

**Objective.** Solve 0/1 knapsack and Max-Cut with exact methods and heuristics, and formulate them as QUBO/Ising. **Baseline:** ratio-greedy and brute force. Code: `trackB_discrete/`.

**Methods.** Greedy, half-approximation greedy, dynamic programming, ILP (OR-Tools SCIP, cross-checked with scipy.optimize.milp), own Branch-and-Bound, brute force; Max-Cut as ILP and QUBO; QUBO to Ising conversion; cardinality constraint with a penalty term; simulated annealing and greedy local search.

**Results (MEASURED).**

- All five exact knapsack solvers agreed on 40 instances (n = 15 and 20). Greedy gap to the optimum: 0.27% (n = 50), 0.01% (n = 200), 0% (n = 1000).
- At n = 1000 (strongly correlated values) my Branch-and-Bound explored about 2.27 million nodes in 2.3 s, against 0.055 s for DP and 0.047 s for the SCIP ILP.
- Max-Cut on 15 graphs (n = 10, 12, 14): brute force, the ILP and the QUBO brute force gave identical optimal cuts. Simulated annealing hit the exact optimum in 84% of runs at n = 14; greedy local search with 20 restarts hit it 100% of the time.
- QUBO and Ising energies agree on every assignment (maximum difference 0). The cardinality penalty gave 0% feasible optima at 0.1x and 0.5x of the largest item value and 100% at 1x and above.

**Limits.** Small graphs, one annealing setting, only two penalty values around the feasibility switch, own Branch-and-Bound in Python. Feature selection on a real dataset and portfolio selection were not run; the selection problem used synthetic items.

## 4. Track C - Quantum optimization (all SIMULATED)

**Objective.** Solve the Track B Max-Cut instances with QAOA and study gradient estimation, shots, noise, VQE, quantum natural gradient, a QNN and barren plateaus. **Baseline:** classical greedy local search, annealing, random assignment. Code: `trackC_quantum_opt/`.

**Results (SIMULATED).**

- My numpy statevector QAOA agrees with PennyLane to about 3e-14. The per-gate parameter-shift gradient matches finite differences; a naive single shift of the shared gamma parameter is wrong on weighted graphs (-0.21 against the correct -19.56).
- Expected-cut ratio (n = 10, p = 1, 2, 3): Adam 0.727/0.788/0.763, COBYLA 0.712/0.790/0.774, SPSA 0.674/0.679/0.666; random assignment gives 0.64 to 0.66. Classical local search and annealing reached the exact optimum (ratio 1.0) on every instance, so shallow QAOA is clearly worse at this size.
- Shot noise: the standard deviation of the estimate follows 1/sqrt(shots) (fitted log-log slope -0.494). With 1000-shot evaluations and a budget of 300 circuit evaluations: Adam with parameter-shift 0.732, COBYLA 0.686 (it stopped after about 38 evaluations), SPSA 0.665, random parameters 0.649 +/- 0.035.
- Depolarizing noise (q = 0.05, three 8-node graphs): ratio fell from 0.860 to 0.840 (p = 1) and from 0.839 to 0.800 (p = 2).
- VQE (4 qubits, 3 ansatz layers): quantum natural gradient converged from all 5 starts (errors 4e-7 Ising, 7e-5 Heisenberg, 18 to 24 iterations), while gradient descent and Adam stalled in 2 of 5 Ising runs. It costs more circuit evaluations per run (7,200 against 4,800). With 2 layers every optimizer stalled on the Heisenberg chain, which points to ansatz expressibility. A fast numpy backend (validated to 1e-16 against PennyLane) cut the sweep from an estimated 70 minutes to 68 s (failure log row 18).
- QNN (12 parameters, make_moons): test accuracy 93.3% +/- 4.6% over 3 seeds, 500,000 circuit evaluations per run.
- Barren plateaus: the variance of the gradient of a global cost fell from 4.5e-2 (n = 2) to 9.5e-7 (n = 10), slope -1.35 per qubit in ln(variance); the local cost fell from 9.1e-2 to 5.3e-3 (slope -0.36). Both fits use 5 points. At n = 6 there was no clear dependence on depth between 1 and 16 layers.

**Limits.** All simulated; the budget of 300 evaluations is too small for SPSA; noise study uses three small graphs and a simple depolarizing model; ratios have a large spread (0.02 to 0.09).

## 5. Track D - Submodular, parallel and distributed optimization

**Objective.** Test greedy selection on a submodular objective, measure parallel speedup and its limits, and compare FedAvg, MGDA and GradNorm with baselines. Code: `trackD_parallel/`.

**Submodular (MEASURED).** Facility location showed 0 violations of diminishing returns in 40,000 checks; a deliberately complementary objective showed 297. Brute force and the ILP agree on 160 instances; greedy reached 0.96 to 1.0 of the optimum (minimum 0.9617, guarantee 0.632). Lazy greedy chose identical sets with 61% to 67% fewer evaluations (about 3x faster). On the complementary objective greedy reached only 0.48 of the optimum, where the guarantee does not apply.

**Parallel (MEASURED, platform-limited).** Speedup of 48 half-second tasks was 1.00, 1.23, 1.99, 2.21 and 3.26 on 1, 2, 4, 8 and 12 processes. A separate check (`tools/check_parallel.py`) showed that one task takes 0.129 s alone but 0.39 s per task with 4 workers and up to 0.63 s with 8, so these speedups are limited by this machine (shared cores, power and memory limits), and the fitted Amdahl and Gustafson serial fractions are not interpretable (failure log row 19). Tiny tasks (5 ms) with a fresh pool gave 0.17x because of Windows process start-up; with a warmed pool 2.40x. Load balancing on 96 heavy-tailed tasks (4 workers): equal split 1.56x, LPT 1.61x, dynamic queue 1.61x. Work stealing is a SIMULATION. Data-parallel logistic regression matched the sequential loss exactly; relative to one persistent worker the speedups are 1.25x (2 workers) and 1.59x (4). Pipeline parallelism reached 1.13x (ideal 2.33x); column-sharded model parallelism 1.63x (2 threads) and 2.88x (4).

**Distributed and multi-task (SIMULATED in one process).** FedAvg with one local step and full participation equals centralized gradient descent within 1e-10. On 10 clients, the final loss gaps to the centralized optimum were below 0.004 in every non-IID setting (largest 0.0033), so no material client drift was observed in this setup; 3 of 10 clients per round used 30% of the communication and reached loss 0.6258 against 0.6193. Two-task regression with loss scales 0.37 and 402 (relative final losses task 1 / task 2 / sum): equal weighting 0.273 / 0.005 / 0.279, MGDA 0.049 / 0.242 / 0.291, GradNorm 0.044 / 0.045 / 0.088. MGDA never increased either loss but converged slowly on the large task.

**Limits.** MGDA and GradNorm chose the largest value of a 3-point learning-rate grid, so their results may be understated; one synthetic problem; 5 seeds. An early FedAvg bug (local steps not equal to E) was found by Codex after Antigravity hit its usage limit (failure log row 20).

## 6. Track E - Main case study: DabbaDash, last-mile delivery in Mumbai

**Process and evidence.** A fleet of capacity-50 vehicles serves N customers (synthetic, clustered around Andheri, Bandra, Powai, Borivali, Dadar and Ghatkopar, depot in Andheri East). The naive "driver order" process is long: at n = 200 it drives 2,612.9 km, against 681.5 km for nearest-neighbour and 514.8 km for Clarke-Wright with local search.

**Formulation.** Capacitated vehicle routing: binary arc variables x_ij, load variables u_i (MTZ constraints), minimise total distance subject to every customer visited exactly once, capacity Q = 50 and routes starting and ending at the depot. Distance is haversine times a circuity factor of 1.3. Assumptions and limits: synthetic data, no time windows. Full formulation in `docs/trackE_formulation.md`, strategy table in `docs/trackE_strategy.md`. Code: `trackE_mumbai_routing/`.

**Methods and results (MEASURED, mean over 5 seeds).**

| n | driver order | nearest-neighbour | Clarke-Wright | CW + local search | OR-Tools (10 s; 30 s at n=200) |
|---|---|---|---|---|---|
| 20 | 266.5 km | 131.2 | 96.9 | 96.5 | 95.4 |
| 50 | 647.4 | 266.7 | 169.0 | 166.9 | 165.8 |
| 100 | 1328.7 | 444.7 | 299.0 | 296.2 | 317.3 |
| 200 | 2612.9 | 681.5 | 517.2 | 514.8 | 540.2 |

The exact ILP proved optimality on all 15 small instances (n = 8, 10, 12). Mean gap to the optimum: nearest-neighbour 10% to 38%, Clarke-Wright 0.7% to 1.1%, Clarke-Wright with local search 0% to 0.7%, OR-Tools 0%. At n = 20 the MTZ ILP proved optimality for only 1 of 5 instances within 120 s. Under these time limits Clarke-Wright with local search beat OR-Tools at n = 100 and 200 while running about 250 to 400 times faster. Parallel runs across seeds were slower than sequential (0.11x to 0.13x) because each task takes about 40 ms and Windows process start-up dominates. The QUBO/QAOA route was not applied to this problem because of its size and constraint encoding cost.

## 7. Track F - Compiler optimization (MiniLang)

**Build.** A mini-compiler for a toy language: lexer, parser, AST, three-address IR, code generator and a register VM, plus an independent tree-walking interpreter (the oracle). Code: `trackF_compiler/`.

**Optimizations.** Constant folding and propagation, dead-code elimination, common subexpression elimination (with store-to-load forwarding), loop-invariant code motion, strength reduction (five passes).

**Formulation.** Pass ordering: choose a sequence of up to 5 passes (with repetition) to minimise the total dynamic instruction count over six benchmarks, subject to unchanged output and a compile-time budget. Methods: fixed order, exhaustive search over the 120 permutations (exact), restricted sequences of length at most 2, and greedy.

**Results (MEASURED).** Dynamic instruction count reduction with the best order: fib 29%, invariant 52%, mixed 60%, primes 29%, redundant 76%, sum 23%. On scaled-down benchmarks: no passes 6,309, fixed order 3,588, exhaustive best order 3,493, greedy 3,504 (25 candidates instead of 120). Constant folding alone shows 0.0% because it only adds the folded constant and leaves the old instructions for dead-code elimination to remove (checked in the IR). Strength reduction looked harmful on two benchmarks (-5.8%, -1.8%); the instruction-count metric cannot show a benefit for shifts.

**Correctness.** 18 tests, plus a differential fuzzer: 300 random programs (289 terminating, 11 deliberate division-by-zero traps) compared with the oracle for every pass and for the fixed order (1,800 comparisons, no mismatch). Expected outputs (5000050000, 832040, 303) were computed independently in plain Python.

**Limits.** Pass-order search ran on scaled-down benchmarks; the 2.6% advantage of putting LICM first is not explained; single run.

## 8. Track G - xv6 operating system: lottery scheduler

**Change.** A lottery scheduler with a `settickets` system call in xv6-riscv (upstream commit e93dabe), selectable with `make SCHED=LOTTERY` (default round-robin stays as the baseline). Each runnable process wins with probability tickets / total tickets; random numbers come from a fixed-seed xorshift32 generator protected by its own spinlock. The kernel patch is `trackG_xv6/lottery_scheduler.patch`.

**OS concepts.** Process scheduling, the scheduler's run-queue scan, context switch (swtch), spinlocks around p->lock. **Objectives:** proportional-share fairness, turnaround time of a short job, variance of CPU share.

**Correctness.** xv6 boots and `usertests` ends with ALL TESTS PASSED on the unmodified kernel (5 min 26 s) and on the lottery build.

**Results (MEASURED on QEMU, one CPU, 100 ms tick, 5 repetitions).**

| Workload | Round-robin | Lottery |
|---|---|---|
| W1 work share for 30/20/10 tickets | 0.334 / 0.333 / 0.334 | 0.459 / 0.434 / 0.107 (target 0.50 / 0.33 / 0.17; standard deviation about 0.17) |
| W2 turnaround of a short job: 10 tickets vs 40 tickets | 99.2 vs 99.6 ticks | 94.0 +/- 27.2 vs 32.4 +/- 11.8 ticks (2.9x faster) |
| W3 Jain fairness index, 8 equal processes | 0.995 | 0.650 (some processes got almost no CPU in 60 ticks) |

**Limits.** Round-robin ignores tickets; lottery gives controllable priority (W2) at the price of short-term unfairness (W3), because only 60 draws per process happen in the window. The lottery numbers come from one run done by hand after the automated harness timed out on its second launch (failure log row 16); QEMU timing is not real hardware timing; four of the eight W3 processes under round-robin consistently got about 12% more work, unexplained. Only one kernel feature was implemented.

## 9. Track H - DBMS optimization (PostgreSQL 18.6)

**Data.** A TPC-H-like synthetic dataset at scale factor 0.1 (supplier 1,000, customer 15,000, part 20,000, orders 150,000, lineitem 590,805), seed 42; 10 weighted queries. **Baseline:** primary keys only; weighted average latency 89.9 ms and 93.5 ms in two runs (heavy queries and the average stable within about 5%).

**Formulation.** Index selection: variables x_c in {0,1} for 18 candidate indexes; minimise the weighted workload cost subject to total index size at most a storage budget B. Methods: greedy by benefit per MB, DP knapsack, assignment ILP (OR-Tools SCIP), compared with the default optimizer with no extra index. Code: `trackH_dbms/`.

**Results (MEASURED).** On the planner cost model, at a 10% budget ILP 422,878, greedy 423,703, DP 428,624; at a 50% budget ILP and greedy tie (419,437) but the ILP uses 24.2 MB against 32.5 MB. Predicted and real planner costs differ by under 0.3%. **Measured latency did not improve:** the baseline is 91.8 ms and every configuration is 95 to 108 ms (speedups 0.85x to 0.97x); the configurations with 12 to 18 indexes were 13% to 15% slower, for an unknown reason. Individual queries improved (Q09 11.5 to 1.9 ms, Q06 27.4 to 9.9 ms, Q02 3.3 to 1.0 ms), but the total did not move because the slowest query, Q04, is untouched by any candidate (likely, untested). Result checksums matched the baseline for every configuration.

**Limits.** No index maintenance cost in the model, so the ILP chose useless indexes; planner cost estimates shift between runs because ANALYZE re-samples; single machine; one run per configuration. This is a negative result: estimated cost falls about 4.7%, measured latency does not.

## 10. Track I - Quantum algorithms and QML versus classical (all SIMULATED)

**Algorithms, hand-written from basic gates** (`trackI_quantum_algos/circuits.py`): Deutsch-Jozsa, Bernstein-Vazirani, Grover, QFT and its inverse, quantum phase estimation, Shor order finding (modular multiplication as a matrix, not a gate-level circuit), plus exact quantum-annealing simulation and a depolarizing noise model.

**Results.** The QFT matrix equals the DFT matrix to 5e-15 (up to 6 qubits); phase estimation recovers an exact phase with probability 1; Grover success follows sin^2((2k+1)theta) and agrees with Qiskit's GroverOperator to 12 digits. Shor on N = 15 and 21 works; N = 21 with a = 5 has zero success probability (its order is 6 and 5^3 = -1 mod 21); simulating N = 21 took about 24 s per case against 0.1 to 0.3 s for N = 15 (MEASURED; cost grows exponentially). Annealing on four n = 10 Max-Cut instances: no instance reached 0.9 ground-state probability for annealing times up to 200 (success probability at T = 200 between 0.04 and 0.09); the embedding lower bounds on Chimera and Pegasus are THEORETICAL counts, not embeddings. Noise: the deeper Grover circuit (84 cx gates) fell below 50% of its noiseless success near p = 0.00096, the shallower one (24 cx) near p = 0.0035.

**QML against classical** (moons / circles / breast cancer, 5 seeds, hyperparameters by cross-validation on training data, parameter-matched MLP included): exact quantum kernel SVM 0.860 / 0.964 / 0.918; shot-based kernels 0.758 / 0.838 / 0.862 (100 shots) and 0.824 / 0.930 / 0.894 (1000 shots); variational classifier 0.808 / 0.732 / 0.858. **The best classical model won all 15 paired comparisons.** On moons the quantum kernel did not improve with more training data (0.74 to 0.81 for 20 to 160 samples, while the RBF SVM stayed at 0.85 to 0.94), which suggests an untuned feature-map bandwidth; this was not tested.

**Algorithmic advantage.** Measured oracle calls follow theory (at N = 16,384 linear search needs 8,192.5 calls on average against about 100 for Grover with restarts), but the exact simulation of Grover at n = 20 took 4.23 s against 0.00022 s for a numpy scan: a classical simulation cannot beat a linear scan, and a real speedup would need fault-tolerant hardware and an oracle cheaper than reading the data.

**Contrast with parallel computing (Track D).** Measured parallel speedups 1.0 to 3.3 on 1 to 12 processes (limited by the platform) against theoretical Grover query reductions of 1.8 to 81 at the same sizes; p processors can give at most a speedup of p, while the quantum reduction assumes a suitable oracle and ideal hardware. `docs/trackI_claims.md` lists every claim with its label and what it does not show.

## 11. Track J - Software engineering: test-suite minimization (sortedcontainers)

**Target.** The open-source project grantjenks/python-sortedcontainers at commit 3ac3586: 2,936 non-blank source lines, 299 tests, 99% line coverage, 1,356 lines and 452 branches executed by the suite. **Profile (MEASURED):** one full run takes about 5.3 s (about 23 s under coverage); the stress tests are about 41% of the time and `test_stress_sortedlist.py` alone 29%; 274 of the 299 tests cover no element that no other test also covers.

**Formulation.** Set cover: variables x_t in {0,1} per test, minimise the sum of test runtimes subject to every covered line and branch staying covered. Methods: greedy (by newly covered elements per second), ILP (OR-Tools SCIP, cross-checked with SciPy MILP), and baselines (the full suite, fastest tests first, random and prefix subsets). Code: `trackJ_software_eng/`; the target is cloned by `setup_target.py` and never modified.

**Results (MEASURED).** The ILP subset has 60 tests and ran about 1.80x faster than the full suite (2.17 s against 3.91 s without coverage, spread 0.10 s; 10.3 s against 18.5 s under coverage); other subsets (58 to 111 tests) ran 1.2x to 1.4x faster. **Cost in correctness:** when every subset was run on the real suite, none preserved full coverage (1,291 to 1,311 of 1,356 lines and 413 to 422 of 452 branches), although all selected tests passed. The coverage model (per-test sets recorded with dynamic contexts in one full-suite run) is therefore not valid for subsets; my first explanation (import-time lines) did not match the missing lines, which were method bodies in sorteddict.py and sortedlist.py. Measuring each test's coverage in isolation would be the fix; it was not done, and mutation testing was not run (see Section 14).

**Limits.** Test times are quantised at 10 ms, so the objective is approximate (the sum of per-test costs exceeds the measured wall time); one machine; 3 repetitions. The result is an honest negative: a measured speedup that does not meet its coverage constraint.

## 12. AI tools, failure log and tool comparison

**Tools.** Claude (Sonnet 5.5): planning, prompt writing, result review, documentation. Antigravity (Gemini 4.8 Flash): code generation for set-up and Tracks A to H (Track D part 2 was left partial when its usage limit ran out). Codex ([TODO: model name shown in Codex]): Track D part 2, Track I and Track J. Dates 2026-10-27 to 2026-11-06. The register is in README.md.

**Tool comparison** (`tool_comparison.md`). The same backpropagation task (2-layer network, softmax cross-entropy, L2) was given to Antigravity and to Claude in a fresh chat and judged by one 9-test file written before either tool answered. Both passed 9/9 on the first attempt with no follow-ups, so there was no measurable difference; limits: one small task, one run, and the tests were written by Claude, one of the tools compared.

**Failure log** (`failure_log.md`, 23 entries). Typical errors: code written for an outdated library (PuLP 4.2 API); wrong theory targets (a gap rate of (1 - 1/kappa) instead of its square, which was my prompt's error); explanations that were untested or wrong (constant folding "has nothing to fold", "implicit spectral regularization", a "memory bandwidth bottleneck"); misleading labels ("Best configuration" with a 0.97x speedup); summaries that left out a specified part or reported tests as passed when they were skipped; tests that checked a function against its own output; a specification error in a test value (chain length). Several were caught only because I ran the scripts myself or compared a claim with the raw data.

## 13. Answers to the final questions

**1. Which process, and what evidence of inefficiency?** The main case study (Track E) is last-mile delivery in Mumbai. The baseline "driver order" route drives 2,612.9 km for 200 customers, against 514.8 km after optimization (-80%), and 681.5 km for the stronger nearest-neighbour baseline.

**2. Objective function and constraints.** Minimise total route distance subject to each customer visited exactly once, vehicle capacity 50, and routes starting and ending at the depot (Section 6). Each other track states its formulation: sets, variables, objective, constraints (set cover in J, index selection with a storage budget in H, pass ordering in F, and so on).

**3. Chapter 6 methods tested and why.** See CHECKLIST.md for the full mapping. Continuous: GD to Adam, Newton, Gauss-Newton, backpropagation, curvature approximations (A). Discrete: greedy, DP, ILP, Branch-and-Bound, QUBO/Ising (B). Quantum: QAOA, VQE, parameter-shift, SPSA, natural gradient, QNN, barren plateaus (C) and the algorithms in I. Submodular, parallel and distributed methods (D). Each was matched to a problem whose variable type, size and constraints suit it; where a method does not fit the main problem (QUBO/QAOA for routing) it was demonstrated on a smaller benchmark and rejected for Track E with a reason.

**4. Optimized solution against the baseline.** Section 6 (routing: -80% against driver order, -24% against nearest-neighbour at n = 200), Section 7 (23% to 76% fewer executed instructions), Section 8 (short job 2.9x faster with more tickets, at the price of fairness), Section 9 (no measured latency gain), Section 11 (1.8x faster test subset, but coverage lost).

**5. Trade-offs, failures, limits.** Lottery scheduling trades fairness for control (Jain index 0.65 against 0.995); QAOA at depth 3 stays below classical local search; quantum kernels lost on every comparison; indexes cut estimated cost but not measured latency; the minimized test suite is faster but loses coverage; parallel speedups are limited by the machine; MGDA is slow on tasks with very different scales. The failure log lists 23 specific errors.

**6. Evidence of correctness and reproducibility.** Fixed seeds and a README command per experiment; independent oracles (AST interpreter, brute force, SciPy and PennyLane cross-checks, finite-difference gradient checks); tests per track (about 190 in total); a fuzzer with 1,800 comparisons; result checksums in the database track; xv6 usertests; raw logs in `results/`; commit history marked `[AI:tool]` or `[MANUAL]` and prompt logs per track.

**7. Adapting to other scale, data, constraints or hardware.** Routing: Clarke-Wright with local search scales to large n; use exact ILP only for about 12 customers or fewer; with time windows or a mixed fleet switch to constraint programming or add the constraints to the local search. Index selection: add a maintenance cost and measure real latency per candidate. Test minimization: record coverage per test in isolation. Parallelism: use warmed pools and coarse tasks; on shared hardware measure the slowdown first. Quantum methods: results depend on simulator size (exponential cost), so hardware or tensor-network simulators are needed beyond about 20 qubits.

**8. Compiler, xv6 and DBMS tracks.** Compiler: five IR passes; I learned that pass interaction matters (folding only pays together with dead-code elimination) and verified correctness with an independent oracle and a fuzzer. xv6: a lottery scheduler; I learned how the scheduler, swtch and p->lock work together, and verified with boot, usertests and benchmarks against round-robin. DBMS: index selection as a knapsack/ILP problem; I learned that a planner cost model can be accurate and still not predict measured latency, and verified that query results are identical (checksums) and plans changed.

**9. Quantum, QML and parallel computing against classical.** Quantum: exact algorithm circuits behave as theory predicts (Grover success formula, QFT equals the DFT, phase estimation), but on a simulator they cost exponentially more than the classical task; QAOA, VQE and kernels were simulated at small size and the best classical model was as good or better in every QML comparison; noise degrades deeper circuits faster; barren plateaus reduce global-cost gradients exponentially. Parallel: measured speedups are real but limited (up to 3.3x on 12 processes here) and overhead can make parallel code slower (0.17x for tiny tasks). Nothing here shows a quantum speedup; no real quantum hardware was used.

**10. AI tools, output differences, errors found.** Claude, Antigravity (Gemini 4.8 Flash) and Codex were used and are listed in README.md; the tool comparison showed no difference on one small task (both 9/9). Codex found a step-count bug in Antigravity's partial FedAvg code that its own tests had missed. The failure log documents 23 errors, for example the outdated PuLP API, an invented explanation of a convergence rate, a harness that failed on its second QEMU launch, and coverage sets that did not transfer to test subsets.

## 14. Limitations and next steps

- Track J: isolate per-test coverage to make the subsets valid, then run mutation testing to measure the loss in fault detection.
- Track H: add an index maintenance cost and re-run the latency comparison on a larger scale factor.
- Track G: a second kernel feature (for example a stride scheduler) and more than 5 repetitions per workload, with the automated harness fixed and run end to end.
- Track I: tune the quantum-kernel bandwidth; run an experiment on real hardware if access becomes available.
- Chapter 6 items not covered: feature selection on a real dataset, portfolio selection and scheduling as QUBO, and a measurement of locality (see CHECKLIST.md).
- All experiments ran on one machine with 3 to 5 repetitions; larger sample sizes would tighten most comparisons.