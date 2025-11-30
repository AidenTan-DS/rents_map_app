import os
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st

from config_data import (
    CBSA_SHP_PATH,
    ZCTA_SHP_PATH,
    CBSA_ZIP_PATH,
    ZCTA_ZIP_PATH,
    MANUAL_CBSA_NAME_MAP,
)
from config_data import compute_rankings


# =========================
# 1. Shapefile loading
# =========================

def _resolve_shapefile_path(shp_path: str, zip_path: str, label: str) -> str:
    """
    Determines which path to use for loading a shapefile.

    Priority:
        1. Use the uncompressed .shp file if it exists.
        2. Otherwise use the .zip file in the same folder.
        3. If both are missing, raise an error.

    Returns a path that can be passed directly to geopandas.read_file:
        - A normal .shp path, or
        - A 'zip://path/to/zipfile.zip'
    """
    # Prefer .shp
    if shp_path and os.path.exists(shp_path):
        return shp_path

    # If not found, use zip
    if zip_path and os.path.exists(zip_path):
        return f"zip://{zip_path}"

    # Nothing found → error
    raise RuntimeError(
        f"{label}: Local shapefile not found. Expected at either "
        f"'{shp_path}' or '{zip_path}'."
    )


@st.cache_resource(show_spinner="🗺️ Loading ZIP code boundaries...")
def load_zcta_shapes() -> gpd.GeoDataFrame:
    """Load ZCTA (ZIP Code Tabulation Area) boundaries."""
    path = _resolve_shapefile_path(ZCTA_SHP_PATH, ZCTA_ZIP_PATH, "ZCTA")
    gdf = gpd.read_file(path)

    if "ZCTA5CE10" not in gdf.columns:
        raise RuntimeError("ZCTA shapefile is missing the column 'ZCTA5CE10'.")

    gdf["zip_code_str"] = gdf["ZCTA5CE10"].astype(str).str.zfill(5)
    return gdf


@st.cache_resource(show_spinner="🏙️ Loading metro area boundaries...")
def load_cbsa_shapes() -> gpd.GeoDataFrame:
    """Load CBSA (Core-Based Statistical Area) boundaries."""
    path = _resolve_shapefile_path(CBSA_SHP_PATH, CBSA_ZIP_PATH, "CBSA")
    gdf = gpd.read_file(path)

    if "NAME" not in gdf.columns:
        raise RuntimeError("CBSA shapefile is missing the column 'NAME'.")

    gdf["name_lower"] = gdf["NAME"].astype(str).str.lower()
    return gdf


# =========================
# 2. City / CBSA matching utilities
# =========================

def parse_city_state(city: str, city_full: str):
    """
    Parse a full city string like 'Seattle, WA'
    into (city_name, state_abbrev).
    """
    raw = city_full or city or ""
    raw = str(raw)
    parts = [p.strip() for p in raw.split(",")]

    if len(parts) >= 2:
        city_part = parts[0]
        state_part = parts[1]
    else:
        city_part = parts[0] if parts else ""
        state_part = ""

    city_base = city_part.strip()
    state_abbrev = state_part.strip().upper()[:2] if state_part else ""
    return city_base, state_abbrev


def build_city_tokens(city_base: str):
    """
    Tokenize a city name to help with fuzzy matching.
    Handles hyphens and similar separators.
    """
    city_base = (city_base or "").strip().lower()
    if not city_base:
        return []

    tokens = [city_base]
    for sep in ["-", "–", "—"]:
        if sep in city_base:
            tokens.extend([t.strip() for t in city_base.split(sep) if t.strip()])

    # Remove duplicates while preserving order
    return list(dict.fromkeys(tokens))


def resolve_manual_cbsa_name(city: str, city_full: str):
    """
    Handle special CBSA matching cases (DC, Boston, etc.)
    """
    key = (city_full or city or "").strip().lower()
    if key in MANUAL_CBSA_NAME_MAP:
        return MANUAL_CBSA_NAME_MAP[key]

    if "boston" in key:
        return "Boston-Cambridge-Newton, MA-NH"

    return None


@st.cache_data
def build_city_cbsa_polygons(
    df_city: pd.DataFrame,
    _cbsa_gdf: gpd.GeoDataFrame,
    metric_name: str,
) -> gpd.GeoDataFrame:
    """
    Given aggregated city-level metrics, match each city to a corresponding CBSA polygon.
    Returns a GeoDataFrame suitable for metro-level choropleths.
    """
    cbsa_gdf = _cbsa_gdf.copy()
    if "name_lower" not in cbsa_gdf.columns:
        cbsa_gdf["name_lower"] = cbsa_gdf["NAME"].astype(str).str.lower()

    # Precompute centroids (EPSG 4326) for nearest-distance fallback
    cbsa_4326 = cbsa_gdf.to_crs(epsg=4326)
    centroids = cbsa_4326.geometry.centroid
    cbsa_gdf["centroid_lat"] = centroids.y
    cbsa_gdf["centroid_lon"] = centroids.x

    cbsa_name_lower = cbsa_gdf["name_lower"]
    cbsa_name_upper = cbsa_gdf["NAME"].astype(str).str.upper()

    records = []

    for _, row in df_city.iterrows():
        city = str(row["city"])
        city_full = str(row.get("city_full", city)).strip()
        avg_value = row["avg_metric_value"]
        lat0 = float(row.get("lat", np.nan))
        lon0 = float(row.get("lon", np.nan))

        if not city_full:
            continue

        candidates = cbsa_gdf.iloc[0:0]

        # 1. Manual override
        manual_name = resolve_manual_cbsa_name(city, city_full)
        if manual_name:
            manual_matches = cbsa_gdf[cbsa_gdf["NAME"] == manual_name]
            if not manual_matches.empty:
                best = manual_matches.iloc[0]
                records.append(
                    {
                        "city": city,
                        "city_full": city_full,
                        "metro_name": city_full,
                        "avg_metric_value": avg_value,
                        "geometry": best.geometry,
                    }
                )
                continue

        # 2. Exact / contains match
        city_full_lower = city_full.lower()
        exact = cbsa_gdf[cbsa_name_lower == city_full_lower]
        if exact.empty:
            contains = cbsa_gdf[cbsa_name_lower.str.contains(city_full_lower, na=False)]
        else:
            contains = exact
        candidates = contains

        # 3. Fuzzy city-base + state matching
        if candidates.empty:
            city_base, state_abbrev = parse_city_state(city, city_full)
            tokens = build_city_tokens(city_base)
            if tokens:
                base_mask = cbsa_name_lower.apply(
                    lambda name: any(t in name for t in tokens)
                )
                if base_mask.any():
                    if state_abbrev:
                        state_mask = cbsa_name_upper.str.contains(state_abbrev, na=False)
                        mask = base_mask & state_mask
                        if mask.any():
                            candidates = cbsa_gdf[mask]
                    else:
                        candidates = cbsa_gdf[base_mask]

        if candidates.empty:
            continue

        # Multiple CBSA matches → pick the geographically closest one
        if (
            len(candidates) > 1
            and np.isfinite(lat0)
            and np.isfinite(lon0)
        ):
            cand = candidates.copy()
            dlat = cand["centroid_lat"] - lat0
            dlon = cand["centroid_lon"] - lon0
            cand["dist2"] = dlat * dlat + dlon * dlon
            cand = cand.sort_values("dist2")
            best = cand.iloc[0]
        else:
            best = candidates.iloc[0]

        records.append(
            {
                "city": city,
                "city_full": city_full,
                "metro_name": city_full,
                "avg_metric_value": avg_value,
                "geometry": best.geometry,
            }
        )

    if not records:
        return gpd.GeoDataFrame(
            columns=["city", "city_full", "metro_name", "avg_metric_value", "geometry"]
        )

    gdf_out = gpd.GeoDataFrame(records, geometry="geometry", crs=cbsa_gdf.crs)
    gdf_out = compute_rankings(gdf_out, "avg_metric_value", "city")
    return gdf_out


# =========================
# 3. Metro → ZIP polygons
# =========================

def get_zip_polygons_for_metro(selected_city, zcta_shapes, df_zip_metric):
    """
    Return ZIP-level polygons and metric values for a given metro.

    Parameters
    ----------
    selected_city : str
        The metro/city selected at the top level.
    zcta_shapes : GeoDataFrame
        ZCTA geographic boundaries.
    df_zip_metric : DataFrame
        Contains ['city', 'zip_code_str', 'metric_value', ...]

    Returns
    -------
    (zip_df_city, gdf_merge)
        zip_df_city : rows of df_zip_metric for this city
        gdf_merge   : GeoDataFrame of ZCTA merged with metrics
    """
    zip_df_city = (
        df_zip_metric[df_zip_metric["city"] == selected_city]
        .dropna(subset=["metric_value"])
        .reset_index(drop=True)
    )
    if zip_df_city.empty:
        return zip_df_city, gpd.GeoDataFrame()

    zip_df_small = (
        zip_df_city[["zip_code_str", "metric_value", "city_full"]]
        .drop_duplicates()
    )

    gdf_merge = zcta_shapes.merge(zip_df_small, on="zip_code_str", how="inner")
    return zip_df_city, gdf_merge
