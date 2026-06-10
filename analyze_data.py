"""
=============================================================================
  Antigravity Corp -- Analytical Queries via DuckDB  (Phase 3)
  Author : Senior Data Engineer
  Date   : 2026-06-10
=============================================================================
  Runs three analytical SQL queries directly on cleaned CSV files using
  DuckDB's in-process engine -- zero external database config required.

    Query 1  -  Weekly KPIs + WoW revenue growth  (LAG window function)
    Query 2  -  Conversion funnel with absolute %  (window function)
    Query 3  -  Worst channel x region checkout drop-off

  Prerequisites:  pip install duckdb
=============================================================================
"""

import os
import duckdb

# ---------------------------------------------------------------------------
# Paths -- DuckDB reads CSVs directly via SQL
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # So relative paths in SQL resolve correctly

ORDERS_CSV   = "cleaned_orders.csv"
PRODUCTS_CSV = "cleaned_products.csv"
USERS_CSV    = "cleaned_users.csv"

# Verify files exist
for f in (ORDERS_CSV, PRODUCTS_CSV, USERS_CSV):
    if not os.path.isfile(f):
        raise FileNotFoundError(
            f"Missing: {f}  --  run clean_data.py first."
        )

# ---------------------------------------------------------------------------
# Helper: pretty-print a query result
# ---------------------------------------------------------------------------
def run_query(con, title: str, sql: str, description: str = "") -> None:
    """Execute SQL, fetch as a pandas DataFrame, and print with formatting."""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  {title}")
    if description:
        print(f"  {description}")
    print(sep)

    df = con.execute(sql).fetchdf()

    # Auto-format numeric columns for readability
    fmt_lines = df.to_string(index=False)
    print(f"\n{fmt_lines}")
    print(f"\n  [{len(df)} row(s) returned]")
    print(sep)


# ===================================================================
#  MAIN
# ===================================================================
def main():
    con = duckdb.connect(database=":memory:")

    print("=" * 80)
    print("  ANTIGRAVITY CORP -- DuckDB ANALYTICAL QUERIES (Phase 3)")
    print("=" * 80)
    print(f"  Engine  : DuckDB {duckdb.__version__} (in-process, zero config)")
    print(f"  Source  : {BASE_DIR}")

    # Quick row counts to confirm data is readable
    for label, csv in [("Orders", ORDERS_CSV),
                       ("Products", PRODUCTS_CSV),
                       ("Users", USERS_CSV)]:
        cnt = con.execute(f"SELECT COUNT(*) FROM '{csv}'").fetchone()[0]
        print(f"  {label:<10s}: {cnt:,} rows")

    # ==================================================================
    #  QUERY 1 : Weekly KPIs + Week-over-Week Revenue Growth
    # ==================================================================
    query1 = f"""
    WITH weekly_kpis AS (
        SELECT
            EXTRACT(YEAR  FROM CAST(o.order_date AS TIMESTAMP)) AS year,
            EXTRACT(WEEK  FROM CAST(o.order_date AS TIMESTAMP)) AS week_num,
            MIN(CAST(o.order_date AS DATE))                     AS week_start,
            MAX(CAST(o.order_date AS DATE))                     AS week_end,
            COUNT(DISTINCT o.order_id)                          AS total_orders,
            SUM(o.quantity * p.base_price)                      AS total_revenue,
            ROUND(SUM(o.quantity * p.base_price)
                  / COUNT(DISTINCT o.order_id), 2)              AS aov
        FROM   '{ORDERS_CSV}'  AS o
        JOIN   '{PRODUCTS_CSV}' AS p
          ON   o.product_id = p.product_id
        WHERE  o.order_stage = 'Purchased'
        GROUP  BY year, week_num
    )
    SELECT
        week_num                                                AS week,
        week_start,
        week_end,
        total_orders,
        CONCAT('$', FORMAT('{{:,.2f}}', total_revenue))         AS total_revenue,
        CONCAT('$', FORMAT('{{:,.2f}}', aov))                   AS avg_order_value,
        CONCAT('$', FORMAT('{{:,.2f}}',
            LAG(total_revenue) OVER (ORDER BY year, week_num)
        ))                                                      AS prev_week_revenue,
        CASE
            WHEN LAG(total_revenue) OVER (ORDER BY year, week_num) IS NULL
                THEN '--'
            ELSE CONCAT(
                ROUND(
                    (total_revenue
                     - LAG(total_revenue) OVER (ORDER BY year, week_num))
                    / LAG(total_revenue) OVER (ORDER BY year, week_num)
                    * 100
                , 1), '%')
        END                                                     AS wow_growth
    FROM   weekly_kpis
    ORDER  BY year, week_num;
    """

    run_query(
        con, "QUERY 1 : WEEKLY KPIs + WEEK-OVER-WEEK REVENUE GROWTH",
        query1,
        "Purchased orders only | Revenue = quantity x base_price | LAG() for WoW"
    )

    # ==================================================================
    #  QUERY 2 : Conversion Funnel
    # ==================================================================
    query2 = f"""
    WITH stage_counts AS (
        SELECT
            order_stage,
            COUNT(DISTINCT order_id) AS unique_events,
            -- Assign a sort key so the funnel is in logical order
            CASE order_stage
                WHEN 'Visited'             THEN 1
                WHEN 'Added to Cart'       THEN 2
                WHEN 'Checkout Initiated'  THEN 3
                WHEN 'Purchased'           THEN 4
            END AS stage_order
        FROM '{ORDERS_CSV}'
        GROUP BY order_stage
    )
    SELECT
        stage_order                                     AS step,
        order_stage                                     AS funnel_stage,
        unique_events,
        -- Absolute conversion % relative to Visited (first stage)
        CONCAT(
            ROUND(
                unique_events * 100.0
                / FIRST_VALUE(unique_events) OVER (ORDER BY stage_order),
            1), '%')                                    AS abs_conversion_pct,
        -- Stage-to-stage drop-off %
        CASE
            WHEN LAG(unique_events) OVER (ORDER BY stage_order) IS NULL
                THEN '--'
            ELSE CONCAT(
                ROUND(
                    unique_events * 100.0
                    / LAG(unique_events) OVER (ORDER BY stage_order),
                1), '%')
        END                                             AS step_conversion_pct,
        CASE
            WHEN LAG(unique_events) OVER (ORDER BY stage_order) IS NULL
                THEN '--'
            ELSE CAST(
                LAG(unique_events) OVER (ORDER BY stage_order) - unique_events
                AS VARCHAR)
        END                                             AS drop_off_count
    FROM  stage_counts
    ORDER BY stage_order;
    """

    run_query(
        con, "QUERY 2 : CONVERSION FUNNEL ANALYSIS",
        query2,
        "Absolute % vs Visited baseline | Step-to-step conversion & drop-off"
    )

    # ==================================================================
    #  QUERY 3 : Worst Channel x Region Checkout Drop-off
    # ==================================================================
    query3 = f"""
    WITH checkout_users AS (
        -- Users who initiated checkout
        SELECT DISTINCT o.user_id, u.region, u.channel
        FROM   '{ORDERS_CSV}'  AS o
        JOIN   '{USERS_CSV}'   AS u  ON o.user_id = u.user_id
        WHERE  o.order_stage = 'Checkout Initiated'
    ),
    purchased_users AS (
        -- Users who completed purchase
        SELECT DISTINCT o.user_id
        FROM   '{ORDERS_CSV}'  AS o
        WHERE  o.order_stage = 'Purchased'
    ),
    dropoff AS (
        SELECT
            c.region,
            c.channel,
            COUNT(*)                                         AS checkout_initiated,
            COUNT(p.user_id)                                 AS purchased,
            COUNT(*) - COUNT(p.user_id)                      AS dropped_off,
            ROUND(
                (COUNT(*) - COUNT(p.user_id)) * 100.0
                / COUNT(*),
            1)                                               AS dropoff_rate_pct
        FROM      checkout_users   AS c
        LEFT JOIN purchased_users  AS p  ON c.user_id = p.user_id
        GROUP BY  c.region, c.channel
    )
    SELECT
        region,
        channel,
        checkout_initiated,
        purchased,
        dropped_off,
        CONCAT(dropoff_rate_pct, '%')                        AS dropoff_rate,
        -- Rank from worst to best drop-off
        RANK() OVER (ORDER BY dropoff_rate_pct DESC)         AS severity_rank
    FROM   dropoff
    ORDER  BY dropoff_rate_pct DESC, dropped_off DESC;
    """

    run_query(
        con, "QUERY 3 : CHECKOUT DROP-OFF BY CHANNEL x REGION",
        query3,
        "Users who initiated checkout but never purchased | Ranked worst-first"
    )

    # ------------------------------------------------------------------
    #  Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("  All queries executed successfully.  No external database required.")
    print(f"{'=' * 80}\n")

    con.close()


if __name__ == "__main__":
    main()
