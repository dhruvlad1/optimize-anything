# Chapter 6 Coverage Checklist

**Student:** Dhruv Lad | **Roll No:** 241070037 | **Batch:** B | **Subject:** Optimization Techniques, Lab Experiment 8

This checklist follows the topic list in the assignment (Section 5), because the class slides were not available when it was written. **Before submission, compare it with the Chapter 6 slides and add or remove rows so that it matches the syllabus taught in class.** Status: Done = implemented and checked; Limited = done at small scale or with a stated weakness; Not done = not covered in this submission. All quantum results are SIMULATED.

| Chapter 6 topic | Method | Experiment | Code location | Result | Status |
|---|---|---|---|---|---|
| Foundations | ML as an optimization problem (parameters, loss, L2) | 2-layer network on make_moons | `trackA_continuous/exp5_backprop_ml.py` | GD and Adam 96% test accuracy; L2 = 0.01 gives 93%, L2 = 0.1 gives 81% | Done |
| Foundations | Continuous vs discrete optimization | Tracks A and B side by side | `trackA_continuous/`, `trackB_discrete/` | Gradient methods on R^d; exact and heuristic methods on {0,1}^n | Done |
| Foundations | Convex vs non-convex landscapes | Quadratic, quartic, Himmelblau, Rosenbrock, Rastrigin | `trackA_continuous/exp1_convex.py`, `exp2_nonconvex.py` | GD gap rate 0.8073 vs theory 0.8100 (convex); Rastrigin never solved (non-convex) | Done |
| Foundations | Local minima, global minima, saddle points | Saddle f = x^2 - y^2 + 0.25 y^4; Himmelblau | `trackA_continuous/exp2_nonconvex.py` | GD escape time 31 to 152 iterations; pure Newton converges to the saddle; Himmelblau basins 3/7/5/5 (GD) | Done |
| Foundations | Basic quantum computing concepts | Hand-written circuits | `trackI_quantum_algos/circuits.py` | See the quantum rows below | Done |
| Continuous | Gradient descent (GD) | Convex and non-convex benchmarks | `trackA_continuous/optimizers.py` | Linear rate (1 - 1/kappa)^2 confirmed | Done |
| Continuous | SGD, mini-batch SGD | Noisy quadratic, finite-sum least squares | `trackA_continuous/exp4_noisy.py` | Noise floor scales with sigma squared; at 50n sample gradients GD about 1e-19, SGD about 1e-6 | Done |
| Continuous | Momentum, Nesterov (NAG) | Convex and sensitivity sweeps | `trackA_continuous/exp1_convex.py`, `exp3_sensitivity.py` | Faster than GD on kappa = 100 quadratic (about 170 vs 660 iterations to 1e-6) | Done |
| Continuous | AdaGrad, RMSProp, Adam | Same benchmarks, 10 random starts | `trackA_continuous/exp1_convex.py`, `exp2_nonconvex.py` | Adam reaches the Rosenbrock minimum in 65% of 20 starts, GD 5% | Done |
| Continuous | Newton (pure and damped) | Quadratic, saddle, least squares | `trackA_continuous/optimizers.py` | One step on a quadratic; converges to the saddle (pure) or escapes it (damped) | Done |
| Continuous | Gauss-Newton and Levenberg-Marquardt | y = a exp(b x) fit | `trackA_continuous/curvature.py`, `exp6_curvature.py` | Newton 5, Gauss-Newton 7, LM 7, GD 405 iterations; GN stalls from a poor start, LM converges | Done |
| Continuous | Backpropagation (distinct from the optimizer) | Hand-written backprop, two independent implementations | `trackA_continuous/compare/` | Both pass 9/9 tests (finite differences, stability, XOR) | Done |
| Continuous | Curvature approximations: Hessian-vector products, Fisher, diagonal and low-rank | HVP check, power iteration, Fisher preconditioning | `trackA_continuous/curvature.py`, `exp6_curvature.py` | HVP error 2.9e-11; top eigenvalue 0.709578; Fisher comparison inconclusive | Limited |
| Continuous | Convergence for convex, strongly convex, non-convex cases | Quadratic, quartic, non-convex set | `trackA_continuous/exp1_convex.py` | Linear on strongly convex; quartic does not reach 1e-6 in 2000 iterations for GD/Momentum/NAG | Done |
| Continuous | Step-size and optimizer tuning | Learning-rate sweeps (16 values) | `trackA_continuous/exp3_sensitivity.py` | Tuned rates stored in `results/trackA_tuned_lr.json`; Rosenbrock sweep ranks progress, not convergence | Limited |
| Continuous | Noisy objectives | Noisy quadratic, mini-batch least squares | `trackA_continuous/exp4_noisy.py` | See SGD row | Done |
| Discrete | Greedy methods | Ratio-greedy and half-approximation greedy for knapsack | `trackB_discrete/knapsack.py` | Gap 0.27% (n = 50), 0% (n = 1000); half-approx fixes the hand-built failure case | Done |
| Discrete | Integer linear programming (ILP) | OR-Tools SCIP, SciPy MILP cross-check | `trackB_discrete/knapsack.py` | Agrees with DP and brute force on 40 instances | Done |
| Discrete | Branch-and-Bound | Own implementation with fractional bound | `trackB_discrete/knapsack.py` | Agrees on 40 instances; about 2.27 million nodes at n = 1000 | Done |
| Discrete | Dynamic programming | Knapsack over capacity | `trackB_discrete/knapsack.py` | Exact reference; 0.055 s at n = 1000 | Done |
| Discrete | Feature selection / combinatorial selection | Cardinality-constrained item selection (synthetic) | `trackB_discrete/qubo.py`, `exp2_qubo.py` | Penalty-QUBO exact for P >= 1x max value; no real feature-selection dataset used | Limited |
| Discrete | QUBO and Ising formulations, binary-to-spin conversion | Max-Cut on 15 graphs | `trackB_discrete/qubo.py` | QUBO and Ising energies identical on every assignment | Done |
| Discrete | Penalty terms for constraints | Cardinality penalty sweep | `trackB_discrete/exp2_qubo.py` | 0% feasible below 1x, 100% feasible from 1x | Done |
| Discrete | Portfolio selection / scheduling as QUBO | Not run | - | - | Not done |
| Quantum | QAOA | Max-Cut, p = 1 to 3 | `trackC_quantum_opt/qaoa_sim.py`, `exp1_qaoa.py` | Ratio 0.67 to 0.81; classical local search 1.0 | Done |
| Quantum | VQE | Ising and Heisenberg chains, 4 qubits | `trackC_quantum_opt/vqe.py`, `exp4_vqe.py` | Natural gradient reaches 4e-7 / 7e-5 error from 5 starts (3 layers) | Done |
| Quantum | Parameterized circuits, quantum neural network | Data-re-uploading classifier | `trackC_quantum_opt/exp5_qnn.py` | 93.3% +/- 4.6% test accuracy, 12 parameters | Done |
| Quantum | Parameter-shift rule | Hand-written per-gate shift, vs finite differences and PennyLane | `trackC_quantum_opt/qaoa_sim.py`, `vqe.py` | Exact to 1e-10; naive shared shift wrong on weighted graphs | Done |
| Quantum | SPSA | QAOA with shot noise | `trackC_quantum_opt/optimizers_q.py`, `exp2_shots.py` | 0.665 vs Adam 0.732 vs COBYLA 0.686 (random 0.649) | Done |
| Quantum | Quantum natural gradient | Hand-written metric tensor, checked against PennyLane | `trackC_quantum_opt/vqe.py`, `exp4_vqe.py` | Converges where GD and Adam stall, at higher circuit cost | Done |
| Quantum | Hybrid quantum-classical methods | QAOA, VQE, QNN with classical optimizers | `trackC_quantum_opt/` | See rows above | Done |
| Quantum | Deutsch-Jozsa | n = 2 to 6, all balanced functions for n <= 3 | `trackI_quantum_algos/circuits.py`, `exp1_algorithms.py` | Probability 1 and 0 as required | Done |
| Quantum | Bernstein-Vazirani | n = 2 to 8 | `trackI_quantum_algos/circuits.py` | Secret recovered with probability 1 | Done |
| Quantum | Grover | n = 2 to 10 qubits | `trackI_quantum_algos/circuits.py`, `exp1_algorithms.py` | Matches sin^2((2k+1)theta); agrees with Qiskit to 12 digits | Done |
| Quantum | Quantum Fourier Transform | n = 1 to 6 | `trackI_quantum_algos/circuits.py` | Equals the DFT matrix to 5e-15 | Done |
| Quantum | Quantum phase estimation | 3 to 8 counting qubits | `trackI_quantum_algos/circuits.py` | Exact phase with probability 1 | Done |
| Quantum | Shor's algorithm | N = 15 and 21, matrix-based oracle | `trackI_quantum_algos/circuits.py` | Works for N = 15; N = 21, a = 5 fails (order 6, a^3 = -1) | Limited |
| Quantum | Quantum annealing | Exact simulation on four n = 10 Max-Cut instances | `trackI_quantum_algos/exp2_annealing.py` | No instance reached 0.9 success for T up to 200 | Limited |
| Quantum | Variational classifiers, quantum kernels | Three datasets, 5 seeds, matched classical models | `trackI_quantum_algos/qml.py`, `exp4_qml.py` | Best classical model won all 15 paired comparisons | Done |
| Quantum | Barren plateaus | Gradient variance vs qubits and depth | `trackC_quantum_opt/exp6_barren.py` | Global cost: slope -1.35 per qubit; local: -0.36 | Done |
| Quantum | Finite-shot measurement cost | Shot scaling, noisy optimization | `trackC_quantum_opt/exp2_shots.py` | Standard deviation slope -0.494 (theory -0.5) | Done |
| Quantum | Hardware noise | Simulated depolarizing noise model (no hardware) | `trackC_quantum_opt/exp3_noise.py`, `trackI_quantum_algos/exp3_noise.py` | Deeper circuits degrade faster | Limited |
| Quantum | Limits of QUBO embedding on annealing hardware | Chain-length lower bounds, Chimera and Pegasus | `trackI_quantum_algos/exp2_annealing.py` | THEORETICAL counts for the 15 instances; no hardware | Limited |
| Submodular | Diminishing returns, testing the assumption | check_submodular on random instances | `trackD_parallel/submodular.py`, `exp1_submodular.py` | 0 violations in 40,000 checks; 297 for the complementary objective | Done |
| Submodular | Greedy selection and its guarantee | Facility location vs brute force and ILP | `trackD_parallel/exp1_submodular.py` | Ratio 0.9617 to 1.0 (bound 0.632); 0.48 where the assumption fails | Done |
| Parallel | Load balancing, makespan | Static, LPT, dynamic on 96 heavy-tailed tasks | `trackD_parallel/exp2_parallel.py` | 1.56x / 1.61x / 1.61x on 4 workers | Done |
| Parallel | Work stealing | Discrete-event SIMULATION | `trackD_parallel/exp2_parallel.py` | Within 0.4% of an ideal dynamic scheduler at p = 4 (simulated) | Limited |
| Parallel | Locality | Not measured | - | - | Not done |
| Parallel | Synchronization, communication overhead | Tiny tasks, pool start-up, data-parallel gradient sums | `trackD_parallel/exp2_parallel.py` | 0.17x for tiny tasks with a fresh pool, 2.40x with a warmed pool | Done |
| Parallel | Data, model and pipeline parallelism | Logistic regression, column-sharded layer, 3-stage pipeline | `trackD_parallel/exp2_parallel.py` | 1.59x (4 workers), 2.88x (4 threads), 1.13x | Done |
| Parallel | Speedup, efficiency, Amdahl and Gustafson | Scaling of 48 tasks on 1 to 12 processes | `trackD_parallel/exp2_parallel.py` | Speedup 3.26x on 12 processes; fits not interpretable (platform slowdown) | Limited |
| Strategy selection | Choose methods by variable type, size, constraints, hardware | Strategy table for the routing case study | `docs/trackE_strategy.md` | One row per method with usable size and limits | Done |
| Distributed | FedAvg | Federated logistic regression, 10 clients | `trackD_parallel/federated.py`, `exp3_federated.py` | Equals centralized GD at E = 1; no material drift observed | Done |
| Multi-task | MGDA and GradNorm | Two-task regression | `trackD_parallel/multitask.py`, `exp4_multitask.py` | GradNorm lowest sum of relative losses (0.088); MGDA slow on the large task | Limited |
| Track E | Main case study (routing) | CVRP: nearest-neighbour, Clarke-Wright, local search, ILP, OR-Tools | `trackE_mumbai_routing/` | -80% distance vs driver order at n = 200 | Done |
| Track F | Compiler optimization | Five IR passes, pass ordering | `trackF_compiler/` | 23% to 76% fewer executed instructions; fuzzed against an oracle | Done |
| Track G | xv6 lottery scheduler | Three workloads, round-robin vs lottery | `trackG_xv6/` | Short job 2.9x faster with more tickets; Jain index 0.65 vs 0.995 | Limited |
| Track H | DBMS index selection | Greedy, DP, ILP vs default optimizer | `trackH_dbms/` | Estimated cost -4.7%, measured latency unchanged | Done |
| Track I | Quantum algorithms and QML | See the quantum rows | `trackI_quantum_algos/` | No quantum advantage found; Grover simulation slower than a scan | Done |
| Track J | Test-suite minimization | Set cover: greedy, ILP, baselines | `trackJ_software_eng/` | 1.80x faster subset, but coverage not preserved | Limited |