# config_data.py
"""
Central place for:
- Global constants (tables, shapefiles, map settings)
- Theme/CSS
- Colorscales
- Data loading (Databricks or local files)
- Metric utilities: PTI, rankings, YoY

If you want to switch from Databricks to local CSV/Parquet later,
you only need to modify:
  - USE_LOCAL_DATA flag
  - _load_all_data_local() function
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

# Only needed if you still use Databricks
#from databricks import sql
#from databricks.sdk.core import Config

# ============================================================
# 1. Global flags
# ============================================================

# Set this to True when you want to load data from local files
# instead of from Databricks.
USE_LOCAL_DATA = True

# Local file paths (you can change these later)
LOCAL_HOUSE_FILE = "data/house_ts_agg.csv"   # or .csv
#LOCAL_ZIP_GEO_FILE = "data/zip_geo.parquet"      # or .csv

# ============================================================
# 2. Constants: tables, shapefiles, map settings
# ============================================================

# Databricks tables (only used when USE_LOCAL_DATA = False)
HOUSE_TABLE = "workspace.data511.house_ts"
ZIP_GEO_TABLE = "workspace.data511.zip_geo"

# Shapefile paths
CBSA_SHP_PATH = "data/cb_2018_us_cbsa_500k.shp"
ZCTA_SHP_PATH = "data/cb_2018_us_zcta510_500k.shp"

# Map center & zoom
US_CENTER_LAT = 39.8283
US_CENTER_LON = -98.5795
US_ZOOM_LEVEL = 3.8

# Map bounds to keep users within the U.S. region when panning
US_BOUNDS = {
    "west": -130,
    "east": -65,
    "south": 22,
    "north": 52,
}

# Manual mapping for special metros → CBSA.NAME (keys are lowercase)
MANUAL_CBSA_NAME_MAP = {
    "dc_metro": "Washington-Arlington-Alexandria, DC-VA-MD-WV",
}

# ============================================================
# 3. Theme CSS & colorscales
# ============================================================

def get_dynamic_css(is_dark_mode: bool) -> str:
    """Generate CSS based on current theme (light / dark)."""
    if is_dark_mode:
        return """
        <style>
        .modebar {display: none !important;}
        .metric-card {
            background: #111827;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            border: 1px solid #1f2937;
            margin-bottom: 0.75rem;
        }
        .stButton > button {
            transition: all 0.2s ease;
            border-radius: 8px;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        .stSelectbox > div > div {
            border-radius: 8px;
        }
        </style>
        """
    else:
        return """
        <style>
        .modebar {display: none !important;}
        .metric-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 2px rgba(15,23,42,0.06);
            margin-bottom: 0.75rem;
        }
        .stButton > button {
            transition: all 0.2s ease;
            border-radius: 8px;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(148,163,184,0.4);
        }
        .stSelectbox > div > div {
            border-radius: 8px;
        }
        </style>
        """

def get_colorscale(metric_type: str, is_dark_mode: bool = False):
    """
    Return a warm, bright colorscale for the given metric.
    Both metrics use warm palettes; PTI is a bit more red-ish,
    price is more gold / orange.
    """
    if "PTI" in metric_type:
        # Peach → orange → red
        return [
            [0.0, "#fff7ed"],   # very light peach
            [0.25, "#fed7aa"],  # soft orange
            [0.5, "#fdba74"],   # mid orange
            [0.75, "#fb923c"],  # vivid orange
            [1.0, "#c2410c"],   # deep warm red-brown
        ]
    else:
        # Light yellow → gold → deep orange
        return [
            [0.0, "#fefce8"],   # very light yellow
            [0.25, "#fde68a"],  # light gold
            [0.5, "#fbbf24"],   # gold
            [0.75, "#f97316"],  # bright orange
            [1.0, "#b45309"],   # deep orange-brown
        ]

# ============================================================
# 4. Databricks SQL helper (only used when USE_LOCAL_DATA = False)
# ============================================================

def _sql_query(query: str) -> pd.DataFrame:
    """Execute SQL query against Databricks SQL warehouse."""
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID is not configured")

    cfg = Config()
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: cfg.authenticate,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()

# ============================================================
# 5. Data loading (Databricks vs local)
# ============================================================

def _standardize_house_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply common cleaning and derived columns:
      - zip_code_str
      - city_clean
      - ensure numeric types
    This function expects columns:
      city, city_full, zip_code, year,
      median_sale_price, per_capita_income, lat, lon
    """
    df = df.copy()
    df["zip_code"] = df["zip_code"].astype("Int64")
    df["zip_code_str"] = (
        df["zip_code"].astype(str)
        .str.replace("<NA>", "", regex=False)
        .str.zfill(5)
    )
    df["city_clean"] = df["city"].astype(str).str.lower().str.strip()
    df["year"] = df["year"].astype(int)
    df["median_sale_price"] = pd.to_numeric(df["median_sale_price"], errors="coerce")
    df["per_capita_income"] = pd.to_numeric(df["per_capita_income"], errors="coerce")
    return df

def _load_all_data_databricks() -> pd.DataFrame:
    """
    Original implementation: query Databricks and aggregate
    to city/zip/year level.
    """
    query = f"""
        SELECT
            h.city,
            h.city_full,
            h.zip_code,
            h.year,
            AVG(h.median_sale_price) AS median_sale_price,
            AVG(h.per_capita_income) AS per_capita_income,
            AVG(g.lat) AS lat,
            AVG(g.lon) AS lon
        FROM {HOUSE_TABLE} h
        LEFT JOIN {ZIP_GEO_TABLE} g
          ON CAST(h.zip_code AS INT) = CAST(g.zip_code AS INT)
        WHERE h.median_sale_price IS NOT NULL
          AND h.median_sale_price > 0
        GROUP BY
            h.city, h.city_full, h.zip_code, h.year
    """
    raw = _sql_query(query)
    return _standardize_house_df(raw)

def _load_all_data_local() -> pd.DataFrame:
    """
    Local loading version.

    You can customize this function later, e.g.:
      - read from a single aggregated Parquet file
      - or read house + zip_geo and join them

    For now, this uses a simple "one aggregated file" approach.
    Make sure LOCAL_HOUSE_FILE contains the columns:
        city, city_full, zip_code, year,
        median_sale_price, per_capita_income, lat, lon
    """

    if LOCAL_HOUSE_FILE.lower().endswith(".parquet"):
        house = pd.read_parquet(LOCAL_HOUSE_FILE)
    else:
        house = pd.read_csv(LOCAL_HOUSE_FILE)

    # If you prefer to also read zip_geo and join:
    # if LOCAL_ZIP_GEO_FILE:
    #     if LOCAL_ZIP_GEO_FILE.lower().endswith(".parquet"):
    #         zip_geo = pd.read_parquet(LOCAL_ZIP_GEO_FILE)
    #     else:
#         zip_geo = pd.read_csv(LOCAL_ZIP_GEO_FILE)
#     zip_geo = zip_geo[["zip_code", "lat", "lon"]].drop_duplicates()
#     house = house.merge(zip_geo, on="zip_code", how="left")

    return _standardize_house_df(house)

@st.cache_data(show_spinner="📊 Loading housing data...")
def load_all_data() -> pd.DataFrame:
    """
    Public data loading function used by app.py.

    It chooses Databricks or local implementation based on
    USE_LOCAL_DATA flag above.

    - When USE_LOCAL_DATA = False:
        data are loaded from Databricks via SQL
    - When USE_LOCAL_DATA = True:
        data are loaded from local files
    """
    if USE_LOCAL_DATA:
        df = _load_all_data_local()
    else:
        df = _load_all_data_databricks()
    return df

# ============================================================
# 6. Metric utilities: PTI, rankings, YoY
# ============================================================

def compute_pti(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Price-to-Income (PTI) ratio for all rows in the input DataFrame.
    Filters out extreme values and rows with missing/invalid data.

    Expects columns:
        median_sale_price, per_capita_income
    """
    df = df[
        df["median_sale_price"].notna()
        & df["per_capita_income"].notna()
        & (df["median_sale_price"] > 0)
        & (df["per_capita_income"] >= 5000)
    ].copy()
    df["PTI"] = df["median_sale_price"] / df["per_capita_income"]
    df.loc[(df["PTI"] < 0.5) | (df["PTI"] > 50), "PTI"] = np.nan
    df = df[df["PTI"].notna()].copy()
    return df

def compute_rankings(df: pd.DataFrame, value_col: str, id_col: str) -> pd.DataFrame:
    """
    Add rank, rank_total, and percentile columns based on value_col.
    Higher values get better ranks (1 = highest).

    - value_col: numeric column to rank by
    - id_col: identifier column (city, ZIP, etc.), not used directly but
              helpful for semantic clarity
    """
    df = df.copy()
    df["rank"] = df[value_col].rank(ascending=False, method="min").astype(int)
    df["rank_total"] = len(df)
    df["percentile"] = ((df["rank_total"] - df["rank"] + 1) / df["rank_total"] * 100).round(1)
    return df

def compute_yoy(
    df_all: pd.DataFrame, current_year: int, group_cols: list, value_col: str
) -> pd.DataFrame:
    """
    Compute year-over-year change and percent change for each group in group_cols.

    Parameters
    ----------
    df_all : DataFrame
        Full dataset across all years.
    current_year : int
        The year you want to compare against previous year.
    group_cols : list
        Columns that define each group (e.g. ["city", "city_full"]).
    value_col : str
        Numeric column used for YoY comparison.
    """
    prev_year = current_year - 1
    df_current = df_all[df_all["year"] == current_year].copy()
    df_prev = df_all[df_all["year"] == prev_year].copy()

    if df_prev.empty:
        df_current["yoy_change"] = np.nan
        df_current["yoy_pct"] = np.nan
        return df_current

    agg_current = df_current.groupby(group_cols, as_index=False).agg({value_col: "mean"})
    agg_prev = df_prev.groupby(group_cols, as_index=False).agg({value_col: "mean"})

    merged = agg_current.merge(
        agg_prev,
        on=group_cols,
        suffixes=("", "_prev"),
        how="left",
    )
    merged["yoy_change"] = merged[value_col] - merged[f"{value_col}_prev"]
    merged["yoy_pct"] = (merged["yoy_change"] / merged[f"{value_col}_prev"] * 100).round(1)
    return merged

@st.cache_data
def get_metro_yoy(df_all_input: pd.DataFrame, current_year: int, metric_type_input: str) -> pd.DataFrame:
    """
    Cached helper to compute metro-level year-over-year changes
    for either PTI or median sale price.

    metric_type_input should be one of:
        - "Price-to-Income Ratio (PTI)"
        - "Median Sale Price"
    """
    df_all_local = df_all_input.copy()
    if metric_type_input == "Price-to-Income Ratio (PTI)":
        df_processed = compute_pti(df_all_local)
        value_col = "PTI"
    else:
        df_processed = df_all_local[df_all_local["median_sale_price"].notna()].copy()
        value_col = "median_sale_price"

    return compute_yoy(df_processed, current_year, ["city", "city_full"], value_col)
