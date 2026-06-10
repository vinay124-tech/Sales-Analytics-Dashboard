"""
=============================================================================
  Antigravity Corp -- AI Insight Engine  (Phase 4)
  Author : Senior AI Engineer
  Date   : 2026-06-10
=============================================================================
  Workflow:
    1. DuckDB query  -> checkout drop-off by region x channel
    2. DataFrame      -> Markdown table string
    3. Gemini 2.5 Flash  -> executive-level insight via google-genai SDK

  Requires:
    pip install duckdb google-genai pandas

  Set your API key:
    set GEMINI_API_KEY=your_key_here        (Windows CMD)
    $env:GEMINI_API_KEY="your_key_here"     (PowerShell)
    export GEMINI_API_KEY=your_key_here     (bash)
=============================================================================
"""

import os
import sys
import time
import textwrap

import duckdb
import pandas as pd
from google import genai
from google.genai import errors as genai_errors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ORDERS_CSV = os.path.join(BASE_DIR, "cleaned_orders.csv")
USERS_CSV  = os.path.join(BASE_DIR, "cleaned_users.csv")

# Model cascade: try primary first, fall back if persistent 503
MODELS     = ["gemini-2.5-flash", "gemini-2.0-flash"]
MAX_RETRIES = 3
SEP        = "=" * 72


# ---------------------------------------------------------------------------
# 1.  DATA FETCHING  --  DuckDB checkout drop-off query
# ---------------------------------------------------------------------------
def fetch_dropoff_data() -> pd.DataFrame:
    """Query cleaned CSVs for checkout-initiated-but-not-purchased combos."""

    print(f"\n{SEP}")
    print("  PHASE 4.1 : DATA FETCHING (DuckDB)")
    print(SEP)

    for f in (ORDERS_CSV, USERS_CSV):
        if not os.path.isfile(f):
            print(f"  [ERROR] Missing: {f}")
            print("  Run clean_data.py first.")
            sys.exit(1)

    con = duckdb.connect(database=":memory:")

    sql = f"""
    WITH checkout_users AS (
        SELECT DISTINCT o.user_id, u.region, u.channel
        FROM   '{ORDERS_CSV}'  AS o
        JOIN   '{USERS_CSV}'   AS u  ON o.user_id = u.user_id
        WHERE  o.order_stage = 'Checkout Initiated'
    ),
    purchased_users AS (
        SELECT DISTINCT o.user_id
        FROM   '{ORDERS_CSV}' AS o
        WHERE  o.order_stage = 'Purchased'
    ),
    dropoff AS (
        SELECT
            c.region,
            c.channel,
            COUNT(*)                                    AS checkout_initiated,
            COUNT(p.user_id)                            AS purchased,
            COUNT(*) - COUNT(p.user_id)                 AS dropped_off,
            ROUND(
                (COUNT(*) - COUNT(p.user_id)) * 100.0
                / COUNT(*), 1
            )                                           AS dropoff_rate_pct
        FROM      checkout_users  AS c
        LEFT JOIN purchased_users AS p ON c.user_id = p.user_id
        GROUP BY  c.region, c.channel
    )
    SELECT
        region        AS "Region",
        channel       AS "Channel",
        checkout_initiated AS "Checkouts",
        purchased          AS "Purchased",
        dropped_off        AS "Dropped Off",
        dropoff_rate_pct   AS "Drop-off %"
    FROM   dropoff
    ORDER  BY dropoff_rate_pct DESC, dropped_off DESC;
    """

    df = con.execute(sql).fetchdf()
    con.close()

    print(f"  Query returned {len(df)} region x channel combinations.")
    print(f"  Worst drop-off : {df.iloc[0]['Region']} / {df.iloc[0]['Channel']}"
          f"  ({df.iloc[0]['Drop-off %']}%)")

    return df


# ---------------------------------------------------------------------------
# 2.  MARKDOWN CONVERSION
# ---------------------------------------------------------------------------
def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to a clean Markdown table string."""

    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for _, row in df.iterrows():
        cells = " | ".join(str(row[c]) for c in cols)
        rows.append(f"| {cells} |")

    md = "\n".join([header, separator] + rows)
    return md


# ---------------------------------------------------------------------------
# 3.  LLM INTEGRATION  --  Google Gemini via google-genai SDK
# ---------------------------------------------------------------------------
def get_gemini_insight(md_table: str) -> str:
    """Send the drop-off data to Gemini and return the insight.

    Implements a model cascade with retry + exponential backoff so that
    transient 503 (overload) errors are handled gracefully.
    """

    print(f"\n{SEP}")
    print("  PHASE 4.2 : LLM INTEGRATION (Google Gemini)")
    print(SEP)

    # ---- API key resolution ----
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(
            "  [ERROR] GEMINI_API_KEY not set.\n"
            "  Set it before running this script:\n"
            "    PowerShell : $env:GEMINI_API_KEY=\"your_key\"\n"
            "    CMD        : set GEMINI_API_KEY=your_key\n"
            "    bash       : export GEMINI_API_KEY=your_key"
        )
        sys.exit(1)

    print(f"  API key    : ...{api_key[-6:]}")
    print(f"  Models     : {' -> '.join(MODELS)}  (cascade)")
    print(f"  Retries    : {MAX_RETRIES} per model (exponential backoff)")

    # ---- Build the prompt ----
    prompt = textwrap.dedent(f"""\
    You are a Senior Revenue Operations Analyst who specialises in exotic
    propulsion and antigravity commerce at Antigravity Corp.

    Below is a data table showing checkout drop-off rates broken down by
    customer acquisition channel and geographic region. Each row represents
    users who initiated a checkout but never completed a purchase.

    {md_table}

    Analyse this data and provide a sharp, executive-level insight in
    EXACTLY 3 sentences:
      - Sentence 1: Identify the single worst drop-off bottleneck
        (region + channel) and quantify the impact.
      - Sentence 2: Explain a plausible root-cause hypothesis, grounded
        in the data patterns you see.
      - Sentence 3: Propose one creative, theme-appropriate solution
        (think antigravity / sci-fi branding) to recover those lost
        conversions.
    """)

    # ---- Call with retry + model fallback ----
    client = genai.Client(api_key=api_key)

    for model_id in MODELS:
        print(f"\n  Trying model: {model_id}")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"    Attempt {attempt}/{MAX_RETRIES} ... ", end="", flush=True)
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                )
                print("OK")
                print(f"  Model used : {model_id}")
                return response.text

            except genai_errors.ServerError as e:
                wait = 2 ** attempt  # 2s, 4s, 8s
                print(f"503 (overloaded)")
                print(f"    Server busy -- waiting {wait}s before retry ...")
                time.sleep(wait)

            except Exception as e:
                print(f"FAILED")
                print(f"    Unexpected error: {e}")
                break  # Don't retry on non-503 errors, try next model

        print(f"  Model {model_id} exhausted all retries.")

    # If we get here, all models failed
    print("\n  [ERROR] All models failed after retries.")
    print("  The Gemini API may be experiencing extended downtime.")
    print("  Please try again in a few minutes.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 4.  MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print(f"\n{SEP}")
    print("  ANTIGRAVITY CORP -- AI INSIGHT ENGINE (Phase 4)")
    print(SEP)

    # Step 1 : fetch data
    df = fetch_dropoff_data()

    # Step 2 : convert to markdown
    md_table = dataframe_to_markdown(df)

    print(f"\n  Markdown table prepared for LLM context:\n")
    for line in md_table.split("\n"):
        print(f"    {line}")

    # Step 3 : get AI insight
    insight = get_gemini_insight(md_table)

    # Step 4 : display result
    print(f"{SEP}")
    print("  GEMINI AI INSIGHT")
    print(SEP)
    print()

    # Word-wrap the insight for clean terminal output
    for paragraph in insight.strip().split("\n"):
        wrapped = textwrap.fill(paragraph.strip(), width=68,
                                initial_indent="  ", subsequent_indent="  ")
        if wrapped.strip():
            print(wrapped)
            print()

    print(SEP)
    print("  Phase 4 complete.  AI-powered insight delivered.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
