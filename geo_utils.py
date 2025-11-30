# geo_utils.py
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st

from config_data import (
    CBSA_SHP_PATH,
    ZCTA_SHP_PATH,
    MANUAL_CBSA_NAME_MAP,
)
from config_data import compute_rankings

@st.cache_resource(show_spinner="🗺️ Loading ZIP code boundaries...")
def load_zcta_shapes() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(ZCTA_SHP_PATH)
    if "ZCTA5CE10" not in gdf.columns:
        raise RuntimeError("ZCTA shapefile missing 'ZCTA5CE10'.")
    gdf["zip_code_str"] = gdf["ZCTA5CE10"].astype(str)
    return gdf

@st.cache_resource(show_spinner="🏙️ Loading metro area boundaries...")
def load_cbsa_shapes() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(CBSA_SHP_PATH)
    if "NAME" not in gdf.columns:
        raise RuntimeError("CBSA shapefile missing 'NAME'.")
    gdf["name_lower"] = gdf["NAME"].astype(str).str.lower()
    return gdf

# ---------- City / CBSA parsing & matching ----------
def parse_city_state(city: str, city_full: str):
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
    city_base = (city_base or "").strip().lower()
    if not city_base:
        return []
    tokens = [city_base]
    for sep in ["-", "–", "—"]:
        if sep in city_base:
            tokens.extend([t.strip() for t in city_base.split(sep) if t.strip()])
    return list(dict.fromkeys(tokens))

def resolve_manual_cbsa_name(city: str, city_full: str):
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
    """Match each city (metro) in df_city to a CBSA polygon."""
    cbsa_gdf = _cbsa_gdf.copy()
    if "name_lower" not in cbsa_gdf.columns:
        cbsa_gdf["name_lower"] = cbsa_gdf["NAME"].astype(str).str.lower()

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

        city_full_lower = city_full.lower()
        exact = cbsa_gdf[cbsa_name_lower == city_full_lower]
        if exact.empty:
            contains = cbsa_gdf[cbsa_name_lower.str.contains(city_full_lower, na=False)]
        else:
            contains = exact
        candidates = contains

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

def get_zip_polygons_for_metro(selected_city, zcta_shapes, df_zip_metric):
    """
    For a given selected_city, return:
    - zip_df_city: metric values for ZIPs in this metro
    - gdf_merge: ZCTA polygons merged with metric values
    """
    zip_df_city = (
        df_zip_metric[df_zip_metric["city"] == selected_city]
        .dropna(subset=["metric_value"])
        .reset_index(drop=True)
    )
    if zip_df_city.empty:
        return zip_df_city, gpd.GeoDataFrame()

    zip_df_small = zip_df_city[["zip_code_str", "metric_value", "city_full"]].drop_duplicates()
    gdf_merge = zcta_shapes.merge(zip_df_small, on="zip_code_str", how="inner")
    return zip_df_city, gdf_merge
