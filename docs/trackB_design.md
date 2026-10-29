# Track B Design: Discrete Optimization & Quadratic Unconstrained Binary Optimization (QUBO)

## 1. The 0/1 Knapsack Problem

The 0/1 Knapsack Problem is a canonical NP-hard combinatorial selection problem. Given $n$ items, each with an integer value $v_i > 0$ and integer weight $w_i > 0$, and a maximum knapsack capacity $W > 0$:

$$\max_{x \in \{0, 1\}^n} \sum_{i=1}^n v_i x_i \quad \text{subject to} \quad \sum_{i=1}^n w_i x_i \le W$$

where decision variable $x_i = 1$ if item $i$ is selected into the knapsack, and $x_i = 0$ otherwise.

### 1.1 Solvers & Algorithmic Approaches
1. **Ratio-Greedy Heuristic (`greedy_ratio`)**:
   Sort items in descending order of value-to-weight ratio $r_i = v_i / w_i$. Greedily insert items as long as remaining capacity permits. While intuitive and fast ($\mathcal{O}(n \log n)$), plain ratio-greedy has an unbounded worst-case approximation ratio (e.g., if a large item blocks two smaller items whose combined value is greater).
2. **1/2-Approximation Algorithm (`greedy_half_approx`)**:
   Returns the best solution between:
   - The greedy ratio selection: $x_{\text{ratio}}$,
   - The single most valuable feasible item: $i^* = \arg\max \{v_i \mid w_i \le W\}$.
   *Theorem*: $\max(v(x_{\text{ratio}}), v_{i^*}) \ge \frac{1}{2} \text{OPT}$.
3. **Dynamic Programming (`dp_knapsack`)**:
   Exact pseudo-polynomial algorithm with time and space complexity $\mathcal{O}(n W)$ using the recurrence:
   $$V(i, c) = \max(V(i-1, c), V(i-1, c - w_i) + v_i)$$
   with backtrack reconstruction to recover the optimal binary selection subset.
4. **Integer Linear Programming (`ilp_knapsack`)**:
   Formulated as a 0/1 integer program solved via Google OR-Tools using the SCIP solver backend, cross-checked against SciPy's `scipy.optimize.milp`.
5. **Branch-and-Bound (`branch_and_bound`)**:
   Depth-first search over the binary decision tree. At each node, items are sorted by $v_i / w_i$. The upper bound on reachable value is computed using the **fractional relaxation (Dantzig bound)**: fill the remaining capacity greedily and allow the critical (break) item to be taken fractionally. Nodes whose upper bound does not exceed the current best incumbent solution are pruned.

---

## 2. The Maximum Cut Problem (Max-Cut)

Given an undirected graph $G = (V, E)$ with edge weights $w_{ij} > 0$, Max-Cut partitions vertices $V$ into two disjoint sets $S$ and $V \setminus S$ to maximize the total weight of cut edges crossing the partition:
$$\text{Cut}(x) = \sum_{(i,j) \in E} w_{ij} (x_i + x_j - 2 x_i x_j) = \sum_{(i,j) \in E} w_{ij} \mathbb{I}(x_i \ne x_j)$$
where $x_i \in \{0, 1\}$ indicates which partition node $i$ belongs to.

### 2.1 Max-Cut ILP Formulation
Introducing edge decision variables $y_{ij} \in \{0, 1\}$ representing cut edges ($y_{ij} = 1 \iff x_i \ne x_j$):
$$\max_{x, y} \sum_{(i,j) \in E} w_{ij} y_{ij}$$
$$\text{subject to} \quad y_{ij} \le x_i + x_j, \quad y_{ij} \le 2 - (x_i + x_j), \quad \forall (i,j) \in E$$
$$x_i \in \{0, 1\}, \quad y_{ij} \in \{0, 1\}$$

---

## 3. QUBO & Ising Formulations

### 3.1 QUBO (Quadratic Unconstrained Binary Optimization)
QUBO expresses combinatorial problems as unconstrained minimization over binary vectors $x \in \{0, 1\}^n$:
$$\min_{x \in \{0, 1\}^n} E_{\text{QUBO}}(x) = x^T Q x = \sum_{i=1}^n Q_{ii} x_i + \sum_{i < j} Q_{ij} x_i x_j$$
where $Q$ is represented as an upper-triangular $n \times n$ matrix (since $x_i^2 = x_i$ for $x_i \in \{0, 1\}$).

#### Max-Cut to QUBO
Since $\text{Cut}(x) = \sum_{(i,j)} w_{ij} (x_i + x_j - 2 x_i x_j)$, maximizing cut corresponds to minimizing $-\text{Cut}(x)$:
$$-\text{Cut}(x) = \sum_{(i,j) \in E} w_{ij} (2 x_i x_j - x_i - x_j)$$
Mapping to upper-triangular $Q$:
- $Q_{ii} = - \sum_{j \in N(i)} w_{ij}$
- $Q_{ij} = 2 w_{ij}$ for $i < j$
Then $E_{\text{QUBO}}(x) = x^T Q x = - \text{Cut}(x)$ (offset = 0).

### 3.2 Ising Spin Model
In statistical physics and quantum annealing, spins take values $s_i \in \{-1, +1\}$.
Using the linear bijection:
$$x_i = \frac{s_i + 1}{2} \iff s_i = 2 x_i - 1$$
Substituting into $x^T Q x$:
$$x_i x_j = \frac{1}{4} (s_i s_j + s_i + s_j + 1), \quad x_i = \frac{1}{2} (s_i + 1)$$
Yielding the standard Ising Hamiltonian:
$$E_{\text{Ising}}(s) = \sum_{i < j} J_{ij} s_i s_j + \sum_{i=1}^n h_i s_i + \text{offset}$$
where:
- $J_{ij} = \frac{1}{4} Q_{ij}$ (for $i < j$)
- $h_i = \frac{1}{2} Q_{ii} + \frac{1}{4} \sum_{j < i} Q_{ji} + \frac{1}{4} \sum_{j > i} Q_{ij}$
- $\text{offset} = \frac{1}{2} \sum_i Q_{ii} + \frac{1}{4} \sum_{i < j} Q_{ij}$

### 3.3 Cardinality-Constrained Selection via Penalty QUBO
To maximize total item value $\sum_{i=1}^n v_i x_i$ subject to selecting exactly $k$ items ($\sum x_i = k$):
We minimize negative value plus a quadratic penalty on constraint violation:
$$\min_{x \in \{0, 1\}^n} - \sum_{i=1}^n v_i x_i + P \left( \sum_{i=1}^n x_i - k \right)^2$$
Expanding the penalty:
$$\left( \sum_{i=1}^n x_i - k \right)^2 = \sum_i x_i^2 + 2 \sum_{i < j} x_i x_j - 2 k \sum_i x_i + k^2 = (1 - 2k) \sum_i x_i + 2 \sum_{i < j} x_i x_j + k^2$$
QUBO upper-triangular matrix:
- $Q_{ii} = - v_i + P (1 - 2k)$
- $Q_{ij} = 2 P$ for $i < j$
- Constant energy offset $= P k^2$.
*Penalty calibration*: If $P$ is too small, infeasible solutions ($|S| \ne k$) may have lower energy than feasible ones; if $P$ is excessively large, numerical precision degrades and energy differences between valid configurations become negligible.

---

## 4. Fair-Comparison Protocol
All benchmark comparisons enforce:
1. **Identical problem instances**: Same $(v, w, W)$ knapsack tuples and $(V, E, w)$ graphs across all solvers.
2. **Deterministic PRNG seeds**: All randomized solvers (Simulated Annealing, greedy restarts) receive explicit integer seeds.
3. **High-resolution timing**: Wall-clock duration recorded via `time.perf_counter()`.
4. **Independent verification**: Every exact solver is compared against independent solvers (`scipy.optimize.milp`, brute force, DP) to assert zero discrepancies.
