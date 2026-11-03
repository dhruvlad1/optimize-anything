# Track D Design: Submodular and Parallel Optimization

This document outlines the theoretical foundations, algorithms, and experimental protocols for Track D:
submodular optimization and parallel computing benchmarks.

All experiments are conducted within `trackD_parallel`. Runtimes, speedups, and efficiency metrics on real tasks
are measured on this machine (Windows, 10 physical cores, 12 logical cores). Work-stealing scheduling across
scaling worker pools is evaluated via discrete-event simulation on measured task duration distributions.

---

## 1. Submodularity and Monotonicity

Let $V = \{0, 1, \dots, n-1\}$ be a finite ground set. A set function $f: 2^V \to \mathbb{R}$ is **submodular** if it
exhibits diminishing returns: for all subsets $A \subseteq B \subset V$ and every element $x \in V \setminus B$,
$$
f(A \cup \{x\}) - f(A) \ge f(B \cup \{x\}) - f(B).
$$
Equivalently, for all $S, T \subseteq V$,
$$
f(S \cup T) + f(S \cap T) \le f(S) + f(T).
$$

A set function $f$ is **monotone non-decreasing** if for all $A \subseteq B \subseteq V$, $f(A) \le f(B)$, which
means all marginal gains are non-negative: $\Delta(x \mid S) = f(S \cup \{x\}) - f(S) \ge 0$.

### Greedy Algorithm and the $(1 - 1/e)$ Bound
For maximizing a monotone submodular function $f$ subject to a cardinality constraint $|S| \le k$ with $f(\emptyset) = 0$,
Nemhauser, Wolsey, and Fisher (1978) proved that the standard greedy algorithm—which starts with $S_0 = \emptyset$ and
iteratively adds $x^* = \arg\max_{x \in V \setminus S} \Delta(x \mid S)$ until $|S| = k$—guarantees:
$$
f(S_{\text{greedy}}) \ge \left(1 - \frac{1}{e}\right) f(S^*) \approx 0.6321 \cdot f(S^*),
$$
where $S^*$ is the optimal subset of size $k$.

### Lazy Greedy (Accelerated Greedy)
Standard greedy evaluates $n - |S|$ marginal gains at each step, totaling $O(k n)$ function evaluations.
**Lazy greedy** (Minoux, 1978) exploits diminishing returns: since $\Delta(x \mid B) \le \Delta(x \mid A)$ for $A \subseteq B$,
the marginal gain of any element from an earlier iteration provides an upper bound on its current marginal gain.
By maintaining candidates in a max-heap keyed by their most recently computed marginal gain:
1. Pop the top element $x$.
2. If its stored marginal gain was computed against the current set $S$, it is guaranteed to be maximal—add $x$ to $S$.
3. Otherwise, recompute $\Delta(x \mid S)$, update its key, and reinsert it into the heap.

Lazy greedy produces the exact same set as standard greedy while typically cutting function evaluations by 80–98%.

---

## 2. Objectives

### Facility Location
Given a non-negative similarity matrix $W \in \mathbb{R}_{\ge 0}^{m \times n}$ where $W_{ij} = \text{sim}(i, j)$ represents
the utility derived by customer $i$ from facility $j$, the facility location objective is:
$$
f(S) = \sum_{i=1}^m \max_{j \in S} W_{ij}, \quad \text{with } f(\emptyset) = 0.
$$
Because the maximum of non-decreasing functions is submodular, $f(S)$ is a canonical monotone submodular function.

**Integer Linear Programming (ILP) Formulation:**
$$
\max \sum_{i=1}^m \sum_{j=1}^n W_{ij} y_{ij}
$$
subject to:
$$
\sum_{j=1}^n x_j = k, \quad y_{ij} \le x_j \; \forall i, j, \quad \sum_{j=1}^n y_{ij} \le 1 \; \forall i, \quad x_j, y_{ij} \in \{0, 1\}.
$$

### Non-Submodular Complementarity Objective
Submodularity requires substitutability (diminishing returns). When elements exhibit **complementarity** (synergy),
marginal gains can increase when another element is present.
We construct:
$$
f_{\text{comp}}(S) = |S| + B \cdot \mathbb{I}(\{0, 1\} \subseteq S),
$$
where bonus $B > 0$. If $A = \emptyset$ and $B' = \{0\}$, the marginal gain of adding item 1 satisfies:
$$
f_{\text{comp}}(\{1\}) - f_{\text{comp}}(\emptyset) = 1,
$$
while
$$
f_{\text{comp}}(\{0, 1\}) - f_{\text{comp}}(\{0\}) = 1 + B.
$$
Since $1 < 1 + B$, diminishing returns is violated. Under a budget constraint $k=2$, if other items $\{2, 3, \dots\}$ offer
higher individual gains than item 0 or 1 alone, greedy picks the non-synergistic items and misses the bonus, achieving
a ratio far below $1 - 1/e$.

---

## 3. Parallel Performance Metrics and Laws

Let $T_1$ be the execution time on 1 worker, and $T_p$ the execution time on $p$ parallel workers.
- **Speedup:** $S(p) = \frac{T_1}{T_p}$
- **Efficiency:** $E(p) = \frac{S(p)}{p} = \frac{T_1}{p T_p}$
- **Makespan:** $M = \max_{w \in \{1, \dots, p\}} t_w$, where $t_w$ is the total elapsed busy time of worker $w$.

### Amdahl's Law (Fixed Workload)
Assuming a fixed problem size with an inherently serial fraction $s \in [0, 1]$ and a parallelizable fraction $1 - s$:
$$
S(p) = \frac{1}{s + \frac{1 - s}{p}} \implies \lim_{p \to \infty} S(p) = \frac{1}{s}.
$$

### Gustafson's Law (Weak Scaling / Scaled Speedup)
Assuming problem size scales with processor count $p$ such that the serial execution time remains constant while
parallel workload grows proportionally:
$$
S_{\text{scaled}}(p) = s + p (1 - s) = p - s(p - 1).
$$

### Karp-Flatt Metric
Given an experimentally measured speedup $S(p)$ on $p$ processors, the experimentally determined serial fraction
(accounting for serial code, scheduling overhead, and communication contention) is:
$$
e = \frac{\frac{1}{S(p)} - \frac{1}{p}}{1 - \frac{1}{p}}.
$$
A constant $e$ across $p$ indicates Amdahl-like algorithmic serial bottlenecks, whereas increasing $e$ indicates
rising communication and synchronization overhead.

---

## 4. Load Balancing and Work Stealing

When tasks have variable or heavy-tailed execution durations $\tau_1, \tau_2, \dots, \tau_N$:
- **Static Equal Split:** Assigns $N/p$ contiguous tasks to each worker. High makespan if heavy tasks cluster in one partition.
- **Longest Processing Time (LPT):** Sort tasks in descending duration order ($\tau_{(1)} \ge \tau_{(2)} \dots$) and greedily
  assign each task to the worker with the smallest accumulated load. LPT achieves makespan $M_{\text{LPT}} \le \left(\frac{4}{3} - \frac{1}{3p}\right) M^*$.
- **Dynamic Scheduling:** Workers pull individual tasks (chunk size 1) from a shared thread-safe queue on demand.
- **Work Stealing (Simulated):** Each worker maintains a local double-ended queue (deque). Workers push/pop local tasks from the bottom.
  When idle, a worker steals tasks from the top of the longest deque. Work stealing bounds idle time with minimal centralized contention.

---

## 5. Parallelism Paradigms

1. **Data Parallelism:** The dataset is partitioned across workers (e.g., sample mini-batches in gradient descent).
   Each worker computes local partial gradients on its shard; the parent synchronizes and sums gradients each iteration.
2. **Model Parallelism:** The model parameters or tensor operations (e.g., linear layer weight matrix $W$) are partitioned
   across workers. Each worker computes an orthogonal slice (e.g., column blocks), followed by concatenation.
3. **Pipeline Parallelism:** Sequential stages of execution run concurrently on distinct workers, communicating via FIFO queues.
   Throughput is bounded by the slowest stage (the bottleneck).

---

## 6. Execution Model: Measured vs. Simulated

- **Measured on host machine:**
  - Facility location and submodular algorithms (Greedy, Lazy Greedy, Brute Force, ILP with OR-Tools SCIP).
  - Multi-process scaling with 1 to 12 workers on CPU-bound NumPy kernels.
  - Scheduling strategies (Static, LPT, Dynamic queue via `multiprocessing.Pool.imap_unordered`).
  - Data parallelism with persistent worker sharding.
  - Pipeline parallelism with `multiprocessing.Queue` processes.
  - Model parallelism with multithreaded column-sharded matrix multiplications releasing the GIL.
- **Simulated:**
  - Discrete-event simulation of work-stealing across $p \in \{2, 4, 8, 16, 32\}$ worker deques on the 96 heavy-tailed task durations.
