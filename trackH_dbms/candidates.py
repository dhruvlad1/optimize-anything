"""trackH_dbms/candidates.py
--------------------------
Hand-written list of 18 candidate secondary indexes for PostgreSQL on the TPC-H
schema, including single-column, composite, and deliberately useless candidates.

SAFETY INVARIANT:
Every index name MUST start with the prefix 'optidx_'. No script may ever drop
or modify indexes that do not match 'optidx_%' (never drop primary keys).
"""

from __future__ import annotations

from typing import Any

PREFIX = "optidx_"

CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "optidx_lineitem_shipdate",
        "table": "lineitem",
        "sql": "CREATE INDEX optidx_lineitem_shipdate ON lineitem (l_shipdate);",
        "rationale": "Accelerates date range scans on l_shipdate in Q01, Q06, and Q07.",
        "useless": False,
    },
    {
        "name": "optidx_lineitem_partkey",
        "table": "lineitem",
        "sql": "CREATE INDEX optidx_lineitem_partkey ON lineitem (l_partkey);",
        "rationale": "Accelerates foreign key joins on l_partkey with the part table.",
        "useless": False,
    },
    {
        "name": "optidx_lineitem_suppkey",
        "table": "lineitem",
        "sql": "CREATE INDEX optidx_lineitem_suppkey ON lineitem (l_suppkey);",
        "rationale": "Accelerates foreign key joins on l_suppkey with the supplier table in Q07.",
        "useless": False,
    },
    {
        "name": "optidx_lineitem_orderkey_shipdate",
        "table": "lineitem",
        "sql": "CREATE INDEX optidx_lineitem_orderkey_shipdate ON lineitem (l_orderkey, l_shipdate);",
        "rationale": "Composite index supporting orderkey joins with shipdate range filtering in Q07.",
        "useless": False,
    },
    {
        "name": "optidx_lineitem_discount_shipdate",
        "table": "lineitem",
        "sql": "CREATE INDEX optidx_lineitem_discount_shipdate ON lineitem (l_discount, l_shipdate);",
        "rationale": "Composite index matching simultaneous range filters on l_discount and l_shipdate in Q06.",
        "useless": False,
    },
    {
        "name": "optidx_lineitem_shipmode",
        "table": "lineitem",
        "sql": "CREATE INDEX optidx_lineitem_shipmode ON lineitem (l_shipmode);",
        "rationale": "Enables index-only or index scan grouping on l_shipmode in Q08.",
        "useless": False,
    },
    {
        "name": "optidx_lineitem_receipt_commit",
        "table": "lineitem",
        "sql": "CREATE INDEX optidx_lineitem_receipt_commit ON lineitem (l_receiptdate, l_commitdate);",
        "rationale": "Composite index on receipt and commit dates for late lineitem inspection in Q10.",
        "useless": False,
    },
    {
        "name": "optidx_orders_custkey",
        "table": "orders",
        "sql": "CREATE INDEX optidx_orders_custkey ON orders (o_custkey);",
        "rationale": "Accelerates customer foreign key joins in Q03, Q04, Q05, Q07 and range filter in Q09.",
        "useless": False,
    },
    {
        "name": "optidx_orders_orderdate",
        "table": "orders",
        "sql": "CREATE INDEX optidx_orders_orderdate ON orders (o_orderdate);",
        "rationale": "Accelerates order date range scans in Q03 and Q05.",
        "useless": False,
    },
    {
        "name": "optidx_orders_custkey_orderdate",
        "table": "orders",
        "sql": "CREATE INDEX optidx_orders_custkey_orderdate ON orders (o_custkey, o_orderdate);",
        "rationale": "Composite index covering customer join and orderdate ordering/filtering in Q03 and Q09.",
        "useless": False,
    },
    {
        "name": "optidx_orders_orderstatus",
        "table": "orders",
        "sql": "CREATE INDEX optidx_orders_orderstatus ON orders (o_orderstatus);",
        "rationale": "Index on low-cardinality order status filter in Q09.",
        "useless": False,
    },
    {
        "name": "optidx_customer_mktsegment",
        "table": "customer",
        "sql": "CREATE INDEX optidx_customer_mktsegment ON customer (c_mktsegment);",
        "rationale": "Index on market segment predicate filter in Q03 and aggregation group in Q04.",
        "useless": False,
    },
    {
        "name": "optidx_customer_nationkey",
        "table": "customer",
        "sql": "CREATE INDEX optidx_customer_nationkey ON customer (c_nationkey);",
        "rationale": "Accelerates customer-to-nation foreign key joins in Q05 and Q07.",
        "useless": False,
    },
    {
        "name": "optidx_part_brand_type",
        "table": "part",
        "sql": "CREATE INDEX optidx_part_brand_type ON part (p_brand, p_type);",
        "rationale": "Composite index matching equality on p_brand and prefix match on p_type in Q02.",
        "useless": False,
    },
    {
        "name": "optidx_part_type",
        "table": "part",
        "sql": "CREATE INDEX optidx_part_type ON part (p_type);",
        "rationale": "Single-column index on part type prefix pattern search in Q02.",
        "useless": False,
    },
    {
        "name": "optidx_supplier_nationkey",
        "table": "supplier",
        "sql": "CREATE INDEX optidx_supplier_nationkey ON supplier (s_nationkey);",
        "rationale": "Accelerates supplier-to-nation foreign key joins in Q07.",
        "useless": False,
    },
    {
        "name": "optidx_customer_acctbal_useless",
        "table": "customer",
        "sql": "CREATE INDEX optidx_customer_acctbal_useless ON customer (c_acctbal);",
        "rationale": "Deliberately useless index: c_acctbal never appears in any query in the workload.",
        "useless": True,
    },
    {
        "name": "optidx_part_size_useless",
        "table": "part",
        "sql": "CREATE INDEX optidx_part_size_useless ON part (p_size);",
        "rationale": "Deliberately useless index: p_size is only projected in Q02 and never used for filtering or joins.",
        "useless": True,
    },
]

# Validation to ensure every index strictly obeys the naming invariant
for cand in CANDIDATES:
    if not cand["name"].startswith(PREFIX):
        raise ValueError(f"Candidate index name {cand['name']!r} must start with {PREFIX!r}")
