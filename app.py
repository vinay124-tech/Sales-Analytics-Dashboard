"""
=============================================================================
  Antigravity Corp -- Sales Command Center  (Phase 5)
  Author : Senior Python Full-Stack Developer
  Date   : 2026-06-10
=============================================================================
  A Streamlit dashboard powered by DuckDB + Plotly + Gemini AI.

  Run:
    $env:GEMINI_API_KEY="your_key"
    streamlit run app.py

  Requires:
    pip install streamlit duckdb pandas plotly google-genai
=============================================================================
"""

import os
import sys
import time
import textwrap

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from google import genai
from google.genai import errors as genai_errors

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ORDERS_CSV   = os.path.join(BASE_DIR, "cleaned_orders.csv")
PRODUCTS_CSV = os.path.join(BASE_DIR, "cleaned_products.csv")
USERS_CSV    = os.path.join(BASE_DIR, "cleaned_users.csv")

# ---------------------------------------------------------------------------
# Gemini config
# ---------------------------------------------------------------------------
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]
MAX_RETRIES   = 3


# ===================================================================
#  PAGE CONFIG
# ===================================================================
st.set_page_config(
    page_title="Antigravity Sales Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===================================================================
#  CUSTOM CSS -- premium dark theme overrides
# ===================================================================
st.markdown("""
<style>
/* ---- Import Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---- Root variables ---- */
:root {
    --bg-primary:    #0a0e17;
    --bg-card:       #111827;
    --bg-card-hover: #1a2235;
    --border-subtle: #1e293b;
    --accent-cyan:   #06b6d4;
    --accent-violet: #8b5cf6;
    --accent-rose:   #f43f5e;
    --accent-amber:  #f59e0b;
    --accent-green:  #10b981;
    --text-primary:  #f1f5f9;
    --text-muted:    #94a3b8;
    --gradient-hero: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 50%, #f43f5e 100%);
}

/* ---- Global ---- */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ---- KPI card ---- */
.kpi-card {
    background: linear-gradient(145deg, #111827 0%, #1a2235 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(6, 182, 212, 0.15);
}
.kpi-card .kpi-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.kpi-card .kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: var(--gradient-hero);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.kpi-card.revenue::before  { background: var(--accent-cyan); }
.kpi-card.aov::before      { background: var(--accent-violet); }
.kpi-card.orders::before   { background: var(--accent-green); }

/* ---- Section headers ---- */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 32px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ---- AI insight box ---- */
.ai-insight-box {
    background: linear-gradient(145deg, #0f1729 0%, #1a1040 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 28px 32px;
    font-size: 1.02rem;
    line-height: 1.75;
    color: #e2e8f0;
    position: relative;
    overflow: hidden;
}
.ai-insight-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #8b5cf6, #06b6d4);
}

/* ---- Plotly chart containers ---- */
.chart-container {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)


# ===================================================================
#  HELPERS
# ===================================================================
@st.cache_resource
def get_duckdb_connection():
    """Persistent DuckDB in-memory connection (cached across reruns)."""
    return duckdb.connect(database=":memory:")


def check_files():
    """Ensure cleaned CSVs exist."""
    missing = [f for f in (ORDERS_CSV, PRODUCTS_CSV, USERS_CSV)
               if not os.path.isfile(f)]
    if missing:
        st.error("**Missing data files.** Run `clean_data.py` first.")
        for m in missing:
            st.code(m)
        st.stop()


def run_sql(sql: str) -> pd.DataFrame:
    """Execute SQL on the cached DuckDB connection."""
    con = get_duckdb_connection()
    return con.execute(sql).fetchdf()


def build_where(regions: list, channels: list) -> str:
    """Build a dynamic SQL WHERE clause from sidebar selections.

    Joins on the users table is handled by the caller; this returns
    filter predicates for u.region and u.channel.
    """
    clauses = []
    if regions:
        in_list = ", ".join(f"'{r}'" for r in regions)
        clauses.append(f"u.region IN ({in_list})")
    if channels:
        in_list = ", ".join(f"'{c}'" for c in channels)
        clauses.append(f"u.channel IN ({in_list})")

    if clauses:
        return "AND " + " AND ".join(clauses)
    return ""


def format_currency(val: float) -> str:
    """$1,234,567.89"""
    return f"${val:,.2f}"


def format_number(val) -> str:
    """1,234"""
    return f"{int(val):,}"


# ===================================================================
#  SIDEBAR
# ===================================================================
def render_sidebar() -> tuple:
    """Render sidebar filters and return (selected_regions, selected_channels)."""

    with st.sidebar:
        st.markdown("## 🎛️ Dashboard Filters")
        st.caption("Narrow the data to specific segments.")

        # Fetch distinct values for dropdowns
        all_regions  = run_sql(f"SELECT DISTINCT region FROM '{USERS_CSV}' ORDER BY region")["region"].tolist()
        all_channels = run_sql(f"SELECT DISTINCT channel FROM '{USERS_CSV}' ORDER BY channel")["channel"].tolist()

        regions = st.multiselect(
            "Region",
            options=all_regions,
            default=all_regions,
            help="Filter orders by customer region",
        )

        channels = st.multiselect(
            "Acquisition Channel",
            options=all_channels,
            default=all_channels,
            help="Filter orders by marketing acquisition channel",
        )

        st.markdown("---")
        st.caption("Antigravity Corp &copy; 2026")

    return regions, channels


# ===================================================================
#  KPI LAYER
# ===================================================================
def render_kpis(where: str):
    """Top row of three KPI cards."""

    sql = f"""
    SELECT
        COALESCE(SUM(o.quantity * p.base_price), 0) AS total_revenue,
        COALESCE(
            SUM(o.quantity * p.base_price)
            / NULLIF(COUNT(DISTINCT o.order_id), 0),
        0) AS aov,
        COUNT(DISTINCT o.order_id) AS total_orders
    FROM   '{ORDERS_CSV}' AS o
    JOIN   '{PRODUCTS_CSV}' AS p ON o.product_id = p.product_id
    JOIN   '{USERS_CSV}'    AS u ON o.user_id    = u.user_id
    WHERE  o.order_stage = 'Purchased'
    {where}
    """
    row = run_sql(sql).iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="kpi-card revenue">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value">{format_currency(row['total_revenue'])}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card aov">
            <div class="kpi-label">Avg Order Value</div>
            <div class="kpi-value">{format_currency(row['aov'])}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card orders">
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value">{format_number(row['total_orders'])}</div>
        </div>
        """, unsafe_allow_html=True)


# ===================================================================
#  CHART 1 : Weekly Revenue Trend
# ===================================================================
def render_weekly_revenue(where: str):
    st.markdown('<div class="section-header">📈 Weekly Revenue Trend</div>',
                unsafe_allow_html=True)

    sql = f"""
    SELECT
        EXTRACT(YEAR  FROM CAST(o.order_date AS TIMESTAMP)) AS year,
        EXTRACT(WEEK  FROM CAST(o.order_date AS TIMESTAMP)) AS week_num,
        MIN(CAST(o.order_date AS DATE))                     AS week_start,
        SUM(o.quantity * p.base_price)                      AS revenue
    FROM   '{ORDERS_CSV}'  AS o
    JOIN   '{PRODUCTS_CSV}' AS p ON o.product_id = p.product_id
    JOIN   '{USERS_CSV}'    AS u ON o.user_id    = u.user_id
    WHERE  o.order_stage = 'Purchased'
    {where}
    GROUP  BY year, week_num
    ORDER  BY year, week_num
    """
    df = run_sql(sql)

    if df.empty:
        st.warning("No purchased orders match the current filters.")
        return

    df["week_label"] = df["week_start"].astype(str)

    fig = px.area(
        df,
        x="week_label",
        y="revenue",
        markers=True,
        labels={"week_label": "Week Starting", "revenue": "Revenue ($)"},
    )
    fig.update_traces(
        line=dict(color="#06b6d4", width=3),
        fillcolor="rgba(6, 182, 212, 0.10)",
        marker=dict(size=8, color="#06b6d4",
                    line=dict(width=2, color="#0a0e17")),
        hovertemplate="<b>Week of %{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(gridcolor="rgba(148,163,184,0.08)", title="Revenue ($)",
                   tickformat="$,.0f"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


# ===================================================================
#  CHART 2 : Conversion Funnel
# ===================================================================
def render_funnel(where: str):
    st.markdown('<div class="section-header">🔻 Conversion Funnel</div>',
                unsafe_allow_html=True)

    sql = f"""
    SELECT
        o.order_stage,
        COUNT(DISTINCT o.order_id) AS unique_events,
        CASE o.order_stage
            WHEN 'Visited'             THEN 1
            WHEN 'Added to Cart'       THEN 2
            WHEN 'Checkout Initiated'  THEN 3
            WHEN 'Purchased'           THEN 4
        END AS stage_order
    FROM '{ORDERS_CSV}' AS o
    JOIN '{USERS_CSV}'  AS u ON o.user_id = u.user_id
    WHERE 1=1
    {where}
    GROUP BY o.order_stage
    ORDER BY stage_order
    """
    df = run_sql(sql)

    if df.empty:
        st.warning("No data matches the current filters.")
        return

    stage_colors = ["#06b6d4", "#8b5cf6", "#f59e0b", "#10b981"]

    fig = go.Figure(go.Funnel(
        y=df["order_stage"],
        x=df["unique_events"],
        textinfo="value+percent initial",
        textfont=dict(family="Inter", size=14),
        marker=dict(color=stage_colors[:len(df)]),
        connector=dict(line=dict(color="rgba(148,163,184,0.2)", width=1)),
        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<br>%{percentInitial:.1%} of Visited<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


# ===================================================================
#  AI INSIGHT ENGINE
# ===================================================================
def get_dropoff_table(where: str) -> pd.DataFrame:
    """Query the worst checkout drop-off segments."""
    sql = f"""
    WITH checkout_users AS (
        SELECT DISTINCT o.user_id, u.region, u.channel
        FROM   '{ORDERS_CSV}' AS o
        JOIN   '{USERS_CSV}'  AS u ON o.user_id = u.user_id
        WHERE  o.order_stage = 'Checkout Initiated'
        {where}
    ),
    purchased_users AS (
        SELECT DISTINCT o.user_id
        FROM   '{ORDERS_CSV}' AS o
        JOIN   '{USERS_CSV}'  AS u ON o.user_id = u.user_id
        WHERE  o.order_stage = 'Purchased'
        {where}
    ),
    dropoff AS (
        SELECT
            c.region,
            c.channel,
            COUNT(*)                          AS checkouts,
            COUNT(p.user_id)                  AS purchased,
            COUNT(*) - COUNT(p.user_id)       AS dropped_off,
            ROUND((COUNT(*) - COUNT(p.user_id)) * 100.0 / COUNT(*), 1) AS dropoff_pct
        FROM      checkout_users  AS c
        LEFT JOIN purchased_users AS p ON c.user_id = p.user_id
        GROUP BY  c.region, c.channel
    )
    SELECT * FROM dropoff ORDER BY dropoff_pct DESC, dropped_off DESC
    """
    return run_sql(sql)


def df_to_markdown(df: pd.DataFrame) -> str:
    """DataFrame -> Markdown table string."""
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep_line = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = " | ".join(str(row[c]) for c in cols)
        rows.append(f"| {cells} |")
    return "\n".join([header, sep_line] + rows)


def call_gemini(md_table: str) -> str:
    """Call Gemini with retry + model cascade."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return ("**GEMINI_API_KEY not set.**\n\n"
                "Set it before launching the app:\n"
                "```\n$env:GEMINI_API_KEY=\"your_key\"\n```")

    prompt = textwrap.dedent(f"""\
    You are a Senior Revenue Operations Analyst specialising in exotic
    propulsion and antigravity commerce at Antigravity Corp.

    Below is a data table of checkout drop-off rates by acquisition
    channel and geographic region:

    {md_table}

    Provide a sharp, executive-level analysis in 2-3 sentences:
    1. Identify the worst bottleneck and quantify the impact.
    2. Hypothesise a root cause based on the data patterns.
    3. Propose one creative, antigravity-themed solution to recover
       those lost conversions.
    """)

    client = genai.Client(api_key=api_key)

    for model_id in GEMINI_MODELS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=model_id, contents=prompt,
                )
                return resp.text
            except genai_errors.ServerError:
                time.sleep(2 ** attempt)
            except Exception as e:
                break
    return "All Gemini models are currently overloaded. Please try again shortly."


def render_ai_section(where: str):
    st.markdown("---")
    st.markdown('<div class="section-header">🧠 AI Revenue Operations Analyst</div>',
                unsafe_allow_html=True)
    st.caption("On-demand executive insight powered by Google Gemini.")

    if st.button("Generate Executive Insight", type="primary",
                 use_container_width=True):
        with st.spinner("Querying data & consulting Gemini ..."):
            df = get_dropoff_table(where)

            if df.empty:
                st.warning("No checkout data for current filters.")
                return

            md_table = df_to_markdown(df)
            insight  = call_gemini(md_table)

        st.markdown(f"""
        <div class="ai-insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View raw drop-off data sent to Gemini"):
            st.dataframe(df, use_container_width=True, hide_index=True)


# ===================================================================
#  MAIN
# ===================================================================
def main():
    # ---- Pre-flight check ----
    check_files()

    # ---- Header ----
    st.markdown("""
    <h1 style="
        text-align: center;
        font-family: Inter, sans-serif;
        font-weight: 800;
        font-size: 2.2rem;
        background: linear-gradient(135deg, #06b6d4, #8b5cf6, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
    ">🚀 Antigravity Sales Command Center</h1>
    <p style="
        text-align: center;
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 28px;
    ">Real-time analytics &bull; DuckDB engine &bull; Gemini AI insights</p>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    regions, channels = render_sidebar()
    where = build_where(regions, channels)

    # ---- KPIs ----
    render_kpis(where)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # ---- Charts ----
    col_left, col_right = st.columns(2)
    with col_left:
        render_weekly_revenue(where)
    with col_right:
        render_funnel(where)

    # ---- AI Insight ----
    render_ai_section(where)


if __name__ == "__main__":
    main()
