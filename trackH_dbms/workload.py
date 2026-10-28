"""
trackH_dbms/workload.py
-----------------------
Defines the benchmark workload: ~10 deterministic queries covering typical
OLAP patterns present in TPC-H.

Each entry in QUERIES is a dict with:
  name   - short identifier
  sql    - parameterless SQL string
  weight - relative frequency weight for weighted-latency metric
"""

from __future__ import annotations

QUERIES: list[dict] = [
    {
        "name": "Q01_lineitem_range",
        "weight": 3,
        "description": "Range filter on l_shipdate — exercises sequential scan vs. index scan.",
        "sql": """
            SELECT
                l_returnflag,
                l_linestatus,
                COUNT(*)            AS count_order,
                SUM(l_quantity)     AS sum_qty,
                SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge
            FROM lineitem
            WHERE l_shipdate <= DATE '1998-09-01'
            GROUP BY l_returnflag, l_linestatus
            ORDER BY l_returnflag, l_linestatus
        """,
    },
    {
        "name": "Q02_part_lookup",
        "weight": 2,
        "description": "Point-like filter on part brand and type — non-key column predicate.",
        "sql": """
            SELECT p_partkey, p_name, p_brand, p_type, p_size, p_retailprice
            FROM part
            WHERE p_brand = 'Brand#13'
              AND p_type LIKE 'STANDARD%'
            ORDER BY p_partkey
        """,
    },
    {
        "name": "Q03_orders_customer_join",
        "weight": 3,
        "description": "Two-table join: orders + customer filtered on mktsegment and orderdate.",
        "sql": """
            SELECT o_orderkey, o_totalprice, o_orderdate, c_name, c_mktsegment
            FROM orders
            JOIN customer ON o_custkey = c_custkey
            WHERE c_mktsegment = 'BUILDING'
              AND o_orderdate >= DATE '1995-01-01'
            ORDER BY o_totalprice DESC
            LIMIT 100
        """,
    },
    {
        "name": "Q04_three_table_agg",
        "weight": 2,
        "description": "Three-table aggregation: lineitem + orders + customer.",
        "sql": """
            SELECT
                c_mktsegment,
                COUNT(DISTINCT o_orderkey) AS num_orders,
                SUM(l_extendedprice * (1 - l_discount)) AS revenue
            FROM lineitem
            JOIN orders   ON l_orderkey = o_orderkey
            JOIN customer ON o_custkey  = c_custkey
            GROUP BY c_mktsegment
            ORDER BY revenue DESC
        """,
    },
    {
        "name": "Q05_four_table_join",
        "weight": 1,
        "description": "Four-table join: lineitem + orders + customer + nation.",
        "sql": """
            SELECT
                n_name,
                SUM(l_extendedprice * (1 - l_discount)) AS revenue
            FROM lineitem
            JOIN orders   ON l_orderkey = o_orderkey
            JOIN customer ON o_custkey  = c_custkey
            JOIN nation   ON c_nationkey = n_nationkey
            WHERE o_orderdate BETWEEN DATE '1994-01-01' AND DATE '1994-12-31'
            GROUP BY n_name
            ORDER BY revenue DESC
        """,
    },
    {
        "name": "Q06_discount_revenue",
        "weight": 2,
        "description": "Single-table revenue calculation with multi-column range predicates.",
        "sql": """
            SELECT SUM(l_extendedprice * l_discount) AS revenue
            FROM lineitem
            WHERE l_shipdate >= DATE '1994-01-01'
              AND l_shipdate <  DATE '1995-01-01'
              AND l_discount BETWEEN 0.05 AND 0.07
              AND l_quantity < 24
        """,
    },
    {
        "name": "Q07_supplier_nation",
        "weight": 1,
        "description": "Supplier-nation join with shipdate range — five-table join.",
        "sql": """
            SELECT
                n1.n_name AS supp_nation,
                n2.n_name AS cust_nation,
                EXTRACT(YEAR FROM l_shipdate) AS ship_year,
                SUM(l_extendedprice * (1 - l_discount)) AS revenue
            FROM supplier
            JOIN lineitem  ON s_suppkey   = l_suppkey
            JOIN orders    ON l_orderkey  = o_orderkey
            JOIN customer  ON o_custkey   = c_custkey
            JOIN nation n1 ON s_nationkey = n1.n_nationkey
            JOIN nation n2 ON c_nationkey = n2.n_nationkey
            WHERE l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
            GROUP BY supp_nation, cust_nation, ship_year
            ORDER BY supp_nation, cust_nation, ship_year
        """,
    },
    {
        "name": "Q08_group_by_shipmode",
        "weight": 2,
        "description": "Grouping on l_shipmode — low-cardinality group key.",
        "sql": """
            SELECT
                l_shipmode,
                COUNT(*) AS num_lines,
                SUM(l_extendedprice) AS total_price
            FROM lineitem
            GROUP BY l_shipmode
            ORDER BY l_shipmode
        """,
    },
    {
        "name": "Q09_custkey_orders_filter",
        "weight": 2,
        "description": "Orders filtered by a specific custkey range — FK column predicate.",
        "sql": """
            SELECT o_orderkey, o_orderdate, o_totalprice
            FROM orders
            WHERE o_custkey BETWEEN 1 AND 200
              AND o_orderstatus = 'O'
            ORDER BY o_orderdate DESC
            LIMIT 200
        """,
    },
    {
        "name": "Q10_late_lineitems",
        "weight": 1,
        "description": "Late-received line items: l_receiptdate > l_commitdate.",
        "sql": """
            SELECT
                l_orderkey,
                l_partkey,
                l_suppkey,
                l_shipdate,
                l_commitdate,
                l_receiptdate
            FROM lineitem
            WHERE l_receiptdate > l_commitdate
            ORDER BY l_orderkey, l_linenumber
            LIMIT 500
        """,
    },
]
