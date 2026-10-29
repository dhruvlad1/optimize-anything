# Track H: DBMS Optimization (Index Selection)

## Overview

This track studies **query-level optimization in PostgreSQL** through controlled
index-selection experiments on a synthetic TPC-H-like dataset.

The central optimization problem is a **knapsack-style index-selection problem**:
given a workload of 10 benchmark queries and a budget of *k* secondary indexes,
choose the subset of candidate indexes that minimizes the weighted average query
latency.

---

## Dataset: Synthetic TPC-H-Like Schema

> [!NOTE]
> Data is generated deterministically with `numpy.random.default_rng(seed=42)`.
> Scale factor 0.1 (approx. 1/10th of TPC-H SF=1).
> The same schema and data are recreated every time `load_data.py` is run.

### Tables and Row Counts

| Table    | Rows    | Description                           |
|----------|---------|---------------------------------------|
| region   | 5       | Geographic regions                    |
| nation   | 25      | Countries referencing region          |
| supplier | 1 000   | Suppliers referencing nation          |
| customer | 15 000  | Customers referencing nation          |
| part     | 20 000  | Parts catalogue                       |
| orders   | 150 000 | Orders referencing customer           |
| lineitem | ~600 000| Line items referencing orders/part/supplier (1–7 per order, avg ~4) |

### Primary and Foreign Keys Only

At baseline, PostgreSQL creates **only primary-key B-tree indexes**.
No additional indexes exist.  The optimization experiments add secondary
indexes one at a time and measure the effect on workload latency.

```
region(r_regionkey) ← nation(n_regionkey)
nation(n_nationkey) ← supplier(s_nationkey), customer(c_nationkey)
customer(c_custkey) ← orders(o_custkey)
part(p_partkey)     ← lineitem(l_partkey)
supplier(s_suppkey) ← lineitem(l_suppkey)
orders(o_orderkey)  ← lineitem(l_orderkey)
lineitem PK: (l_orderkey, l_linenumber)
```

---

## Baseline PostgreSQL Configuration

The script `run_baseline.py` writes the live settings to
`results/trackH_environment.json` at run time.  Key settings of interest:

| Parameter               | Default (typical) | Impact                              |
|-------------------------|-------------------|-------------------------------------|
| `shared_buffers`        | 128 MB            | Buffer pool size                    |
| `work_mem`              | 4 MB              | Per-sort / per-hash memory          |
| `effective_cache_size`  | 4 GB              | Planner cache estimate              |
| `random_page_cost`      | 4.0               | Cost of a random I/O (lowers → more index scans) |
| `seq_page_cost`         | 1.0               | Cost of a sequential page read      |
| `default_statistics_target` | 100          | ANALYZE histogram resolution        |

> [!IMPORTANT]
> These defaults are deliberately left unchanged for the baseline.
> Any tuning of `postgresql.conf` is out of scope for this track.

---

## Optimization Problem Statement

### Informal

Given the 10-query workload `W` and a maximum of `k` secondary indexes,
choose which columns to index so that the weighted average query latency is
minimized.

### Formal (Knapsack Index Selection)

**Sets**

- `Q` — set of queries `{q₁, …, q₁₀}`
- `C` — set of candidate (table, column) pairs that appear in WHERE / JOIN / ORDER BY clauses

**Parameters**

- `wᵢ` — integer weight of query `qᵢ` (frequency proxy)
- `Lᵢ(S)` — median latency (ms) of query `qᵢ` when index set `S ⊆ C` is active
- `k` — maximum number of secondary indexes allowed (budget)

**Decision variable**

- `xⱼ ∈ {0, 1}` — 1 if candidate index `j ∈ C` is created, 0 otherwise

**Objective**

Minimize the weighted average latency:

```
minimize   Σᵢ  wᵢ · Lᵢ(S)   /   Σᵢ wᵢ
subject to Σⱼ xⱼ ≤ k,   xⱼ ∈ {0, 1}
```

Because `Lᵢ(S)` is a black-box function evaluated by running the query through
PostgreSQL, this is a **combinatorial black-box optimization** problem.  The
search space has `2^|C|` configurations.

---

## Index Selection Optimization Models and Solvers

### Problem Formulation: Storage-Constrained Index Selection

Let $C$ be the set of candidate indexes, and $Q$ be the set of workload queries.
Each query $q \in Q$ has frequency/weight $f_q \ge 0$.
Each candidate index $c \in C$ consumes disk storage $\text{size}_c > 0$ bytes (or MB).
We are given a storage budget $B > 0$.

**Decision variables:**
- $x_c \in \{0, 1\}$ for each candidate index $c \in C$, where $x_c = 1$ if index $c$ is created in the database and $0$ otherwise.
- The selected set of indexes is $S = \{c \in C \mid x_c = 1\}$.

**Objective:**
Minimise the workload cost:
$$\min_{S \subseteq C} \sum_{q \in Q} f_q \cdot \text{cost}_q(S)$$

**Constraint:**
Disk storage budget:
$$\sum_{c \in C} \text{size}_c \cdot x_c \le B$$

---

### Solvers and Cost Models

Because evaluating the true query cost $\text{cost}_q(S)$ via PostgreSQL's query optimizer or actual execution requires building index configurations on disk ($2^{|C|}$ combinations), offline index selection relies on proxy mathematical models of index benefit. We implement and evaluate three distinct solvers:

#### (a) Greedy by Benefit per MB (Additive Model)

The additive benefit model assumes that the benefit of a set of indexes is the sum of their individual benefits measured in isolation.
For each candidate index $c \in C$, its total workload benefit is:
$$\text{benefit}_c = \sum_{q \in Q} f_q \cdot \left(\text{cost}_q(\text{none}) - \text{cost}_q(\{c\})\right)$$
where $\text{cost}_q(\text{none})$ is the planner estimated cost with no secondary indexes, and $\text{cost}_q(\{c\})$ is the estimated cost when only index $c$ is present.

- We filter candidates with $\text{benefit}_c > 0$ and compute the ratio:
  $$r_c = \frac{\text{benefit}_c}{\text{size}_c}$$
- Candidates are sorted in descending order of $r_c$ (with candidate name tie-breaking).
- The greedy algorithm iterates through the sorted list, adding index $c$ to the selected set $S$ whenever $\sum_{i \in S \cup \{c\}} \text{size}_i \le B$.
- Time complexity: $O(|C| \log |C|)$.

#### (b) Dynamic Programming (DP) Knapsack (Additive Model)

Using the exact same additive benefit values $\text{benefit}_c$, this solver treats index selection as a classic 0/1 Knapsack Problem:
$$\max \sum_{c \in C} \text{benefit}_c \cdot x_c \quad \text{s.t.} \quad \sum_{c \in C} w_c \cdot x_c \le W, \quad x_c \in \{0, 1\}$$

- **Rounding Scheme:**
  Because dynamic programming requires integer capacities and weights, candidate sizes are rounded **UP** to whole megabytes:
  $$w_c = \max\left(1, \left\lceil \frac{\text{size}_c}{1024 \times 1024} \right\rceil\right)$$
  (or $\lceil \text{size}_c \rceil$ if sizes are already in integer MB units).
  The budget $B$ is rounded **DOWN** to whole megabytes:
  $$W = \left\lfloor \frac{B}{1024 \times 1024} \right\rfloor$$
  Because $\sum_{c \in S} \frac{\text{size}_c}{1024^2} \le \sum_{c \in S} w_c \le W \le \frac{B}{1024^2}$, rounding sizes UP and the budget DOWN guarantees that the actual storage budget constraint $\sum_{c \in S} \text{size}_c \le B$ is never violated.
- Standard 1D table dynamic programming with boolean decision backtracking recovers the exact optimal set $S^*$ under the additive model.
- Time complexity: $O(|C| \cdot W)$, space complexity $O(|C| \cdot W)$.

#### (c) Integer Linear Programming (ILP) with Query Assignment

The additive model overestimates benefits when two candidate indexes both improve the same query (e.g. two alternative indexes on the same table). To capture **substitution**, the assignment ILP explicitly models which single index is chosen by the optimizer for each query:

- **Variables:**
  - $x_c \in \{0, 1\}$ for each candidate index $c \in C$.
  - $y_{q,c} \in \{0, 1\}$ indicating whether query $q$ is assigned to use index $c$.
  - $y_{q,0} \in \{0, 1\}$ indicating whether query $q$ is executed with no secondary index (baseline plan).
- **Constraints:**
  1. *Single assignment:* Each query is assigned to exactly one option:
     $$y_{q,0} + \sum_{c \in C} y_{q,c} = 1 \quad \forall q \in Q$$
  2. *Linking constraint:* A query can only be assigned to an index if that index is selected:
     $$y_{q,c} \le x_c \quad \forall q \in Q, \forall c \in C$$
  3. *Storage budget constraint:*
     $$\sum_{c \in C} \text{size}_c \cdot x_c \le B$$
- **Objective:**
  $$\min \sum_{q \in Q} f_q \left( \text{cost}_{q,\text{none}} \cdot y_{q,0} + \sum_{c \in C} \text{cost}_{q,c} \cdot y_{q,c} \right)$$
- Solved using Google OR-Tools with the SCIP backend, cross-checked against `scipy.optimize.milp`.
- **Capture and Limitations:**
  The assignment ILP accurately captures *substitution* (diminishing returns between competing indexes on the same query). However, it does *not* capture *complementary* index usage, where a single query plan employs multiple indexes simultaneously (e.g., BitmapAnd across two separate indexes on the same table, or independent indexes used on different branches of a multi-table join).

---

### Inherent Modeling Limitations & Real-Cost Verification

None of the proxy models (additive greedy, additive DP, or assignment ILP) fully captures true index interactions:
1. **Complementary index interactions:** A complex join query (e.g. Q04, Q05, Q07) can use an index on `orders(o_custkey)` AND an index on `lineitem(l_orderkey, l_shipdate)` simultaneously in different join subtrees. The assignment model conservatively attributes only the benefit of the single best index.
2. **Index-merge / Bitmap scans:** PostgreSQL's Bitmap Index Scan can combine multiple single-column indexes using BitmapAnd/BitmapOr.
3. **Data-page caching and cache competition:** Real execution latency depends on buffer hit ratios and whether smaller indexes fit better in PostgreSQL's `shared_buffers` or OS page cache.

Therefore, proxy models are used solely to select candidate sets under budget constraints. The **real cost is measured afterwards** by physically creating the selected indexes in PostgreSQL, running `ANALYZE`, and recording:
- The planner's estimated total cost (`EXPLAIN FORMAT JSON`),
- The true median wall-clock execution latency over 5 runs (`time.perf_counter`),
- MD5 checksums of query result sets to guarantee semantic equivalence.

---

## Candidate Indexes

Columns appearing in workload predicates (to be refined after baseline analysis):

| Candidate | Column             | Query beneficiaries          |
|-----------|--------------------|------------------------------|
| idx_1     | lineitem.l_shipdate | Q01, Q06, Q07               |
| idx_2     | part.p_brand       | Q02                          |
| idx_3     | customer.c_mktsegment | Q03, Q04                  |
| idx_4     | orders.o_orderdate | Q03, Q05                     |
| idx_5     | orders.o_custkey   | Q09                          |
| idx_6     | lineitem.l_receiptdate | Q10                      |
| idx_7     | orders.o_orderstatus | Q09                        |
| idx_8     | lineitem.l_discount | Q06                         |

---

## Metrics

| Metric                    | Description                                         |
|---------------------------|-----------------------------------------------------|
| `median_latency_ms`       | Median of 5 timed executions (ms)                   |
| `planning_time_ms`        | EXPLAIN ANALYZE planning time (ms)                  |
| `execution_time_ms`       | EXPLAIN ANALYZE execution time (ms)                 |
| `estimated_total_cost`    | Planner's estimated total cost (arbitrary units)    |
| `shared_hit_blocks`       | Buffer cache hits reported by top plan node         |
| `shared_read_blocks`      | Disk reads reported by top plan node                |
| `plan_node_types`         | Pipe-separated list of node types in the plan tree  |

---

## Known Limitations

- Row counts are small (scale factor ~0.01) — results may not generalize to
  production-scale datasets.
- Latency measurements on a local single-machine PostgreSQL instance include
  OS scheduling noise; 5 timed runs and median aggregation partially mitigate
  this.
- `l_receiptdate` and `l_commitdate` are generated independently (no
  business-rule ordering), so Q10 ("late lineitems") may return a large
  fraction of rows.
- COPY-based bulk loading with psycopg3 requires the data directory to exist
  and be writable; it is excluded from git via `.gitignore`.
