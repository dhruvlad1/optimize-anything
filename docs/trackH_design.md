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

### Candidate Indexes

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
