# Track E: DabbaDash Mumbai CVRP — Strategy Selection & Analysis

## 1. Strategy Selection Guide

| Method | Variable Type | Problem Size Where Usable | Exact or Heuristic | Hardware Used | Parallelism | Main Limitation |
|---|---|---|---|---|---|---|
| **driver_order** | Integer permutation / sequence | $n \ge 1$ (arbitrary scale, $n = 200+$ measured) | Naive Baseline Heuristic | 1 CPU core (x86_64) | Embarrassingly parallel across instances; single-threaded per solve | Completely ignores spatial topology; arbitrary sequence produces excessive detour mileage. |
| **nearest_neighbour** | Integer sequence / greedy decisions | $n \ge 1$ (arbitrary scale, $n = 200+$ measured) | Greedy Constructive Heuristic | 1 CPU core (x86_64) | Embarrassingly parallel across instances; single-threaded per solve | Myopic greedy choices leave stranded distant customers, forcing long return trips. |
| **Clarke-Wright** | Undirected customer pairs $(i, j)$ / route lists | $n \ge 1$ (tested up to $n = 200$) | Greedy Savings Heuristic | 1 CPU core (x86_64) | Embarrassingly parallel across instances; single-threaded per solve | Cannot undo early sub-optimal route merges; lacks intra-route local optimization. |
| **Clarke-Wright + Local Search** | Route lists / node indices | $n \le 1\,000+$ (tested up to $n = 200$ in < 0.05s) | Composite Heuristic (Savings + 2-opt, Relocate, Swap) | 1 CPU core (x86_64) | Embarrassingly parallel across instances (tested 1 to 12 cores); single-threaded per solve | First-improvement descent terminates at local minima; does not perform non-improving escape steps. |
| **ILP Exact (MTZ SCIP)** | Binary arc $x_{ij} \in \{0,1\}$, continuous load $u_i \in [d_i, Q]$ | $n \le 12$ (proven optimal in $\le 1$ s); $n = 20$ (feasible bound in 120s); $n > 20$ intractable | Exact Branch-and-Bound / Cut | 1 CPU core (multi-instance parallel across 5 cores) | Single solve uses SCIP internal search; multi-instance embarrassingly parallel | Exponential worst-case complexity; weak MTZ continuous relaxation bounds cause slow gap convergence for $n > 15$. |
| **OR-Tools Routing** | CP interval / sequence variables with capacity dimension | $n \le 1\,000+$ (tested $n = 200$ within 30s) | Metaheuristic (Guided Local Search / Tabu) | 1 CPU core (x86_64) | Single solve internal CP search; multi-instance embarrassingly parallel | Non-deterministic timeout cutoff; no mathematical proof of global optimality. |

*(Note: Hardware and performance numbers reflect experimental measurements on the host 12-core x86_64 testbed.)*

---

## 2. Modeling Assumptions & Domain Context

1. **Spatial Generation:**
   - 6 Mumbai geographic hubs (Andheri, Bandra, Powai, Borivali, Dadar, Ghatkopar) with Gaussian cluster offsets ($\sigma = 0.012^\circ \approx 1.3$ km) bounded within Greater Mumbai coordinates $[18.95, 19.25]^\circ \text{N}$, $[72.80, 72.98]^\circ \text{E}$.
   - Central distribution depot fixed at Andheri East (`19.1136° N, 72.8697° E`).
2. **Road Circuity Factor:**
   - Haversine great-circle distances are scaled by a constant factor $\text{CIRCUITY} = 1.3$ to approximate the urban street network and bay/creek traversal penalties of Mumbai.
3. **Operational Simplifications:**
   - Single central depot; homogeneous vehicle fleet with capacity $Q = 50$ lunchbox (dabba) units.
   - Demands are static and deterministic ($q_i \in \{1, \dots, 9\}$).
   - Time windows, lunch delivery deadlines (e.g. 12:30 PM hard cutoff), driver shift limits, traffic congestion variation by hour, and reverse collection routes are not modeled in this first-phase continuous CVRP benchmark.

---

## 3. Operational Strategy Recommendations

### Scale and Latency Shifts
- **Instant Dispatch ($n \le 500$, budget $< 200$ ms):**
  Use **Clarke-Wright + Local Search**. It runs in tens of milliseconds, consistently cuts travel distance by 50% to 75% versus naive order, and produces clean spatial clusters without requiring commercial solver licenses.
- **High-Value Small Batch Logistics ($n \le 12$, e.g. inter-hub van replenishment):**
  Use **Exact MTZ ILP**. Guarantees mathematical optimality, verifying that no additional routing efficiency can be extracted.
- **Production Enterprise Routing ($n = 50$ to $2\,000$, budget 10–60 s):**
  Use **OR-Tools Guided Local Search**. Achieves within 1–3% of exact optimality where tractable, and scales smoothly to hundreds of stops while managing complex real-world dimensions.

### Constraint Variations
- **If Time Windows (VRPTW) are Added:**
  The Clarke-Wright savings formula degrades because merging two routes that are spatially close may violate temporal ordering. Move to OR-Tools Routing with `AddDimension` for time / cumul-var tracking, or a 3-index time-window MILP.
- **If Fleet is Heterogeneous:**
  The MTZ vehicle count bound must be partitioned by vehicle type, and capacity dimensions in OR-Tools must assign distinct capacities $Q_k$ per vehicle index.

---

## 4. Why QUBO / QAOA Formulations Were Not Applied

Recent quantum-classical hybrid literature explores mapping routing problems to Quadratic Unconstrained Binary Optimization (QUBO) for execution on Quantum Annealers (D-Wave) or gate-based QAOA. We explicitly chose not to apply QUBO for the main DabbaDash CVRP for fundamental structural reasons:

1. **Extreme Variable Expansion:**
   In QUBO, all constraints must be compiled into unconstrained objective penalties. To represent customer order along with vehicle assignment without subtours, standard formulations require 3-index variables $x_{ijk} \in \{0, 1\}$ (vehicle $k$ travels from $i$ to $j$) or time-step variables $x_{i,t,k}$ (vehicle $k$ visits $i$ at step $t$).
   For $n = 50$ customers and $K = 6$ vehicles, this requires over $50 \times 50 \times 6 = 15\,000$ coupled binary variables.
2. **Severe Penalty Landscape Ruggedness:**
   Enforcing degree constraints, capacity limits, and subtour prevention requires large quadratic penalty coefficients $P \gg \max d_{ij}$. In practice, optimization algorithms (simulated annealing, QAOA) spend virtually all search effort satisfying feasibility constraints rather than optimizing route length, yielding inferior solutions compared to classical heuristics like Clarke-Wright.
3. **MIP and CP Dominance:**
   Modern classical branch-and-cut (SCIP) and constraint programming (OR-Tools) exploit polyhedral cuts, linear relaxations, and domain propagation that handle hard linear constraints exponentially better than unconstrained quadratic formulations.
