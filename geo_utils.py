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
    优先使用未压缩 .shp，如果没有，再用同目录下的 .zip。
    返回可以传给 geopandas.read_file 的路径：
      - 直接 .shp 路径，或者
      - 'zip://data/xxx.zip'
    """
    # 优先用 .shp
    if shp_path and os.path.exists(shp_path):
        return shp_path

    # 其次用 zip
    if zip_path and os.path.exists(zip_path):
        # GeoPandas 支持直接读取 'zip://path/to/zip'
        return f"zip://{zip_path}"

    # 两个都不存在，报错
    raise RuntimeError(
        f"{label}: 找不到本地 shapefile，"
        f"预期位置：'{shp_path}' 或 '{zip_path}'。"
    )


@st.cache_resource(show_spinner="🗺️ Loading ZIP code boundaries...")
def load_zcta_shapes() -> gpd.GeoDataFrame:
    """加载 ZCTA（ZIP Code Tabulation Area）边界。"""
    path = _resolve_shapefile_path(ZCTA_SHP_PATH, ZCTA_ZIP_PATH, "ZCTA")
    gdf = gpd.read_file(path)

    # 确认列名
    if "ZCTA5CE10" not in gdf.columns:
        raise RuntimeError("ZCTA shapefile 缺少 'ZCTA5CE10' 列。")

    gdf["zip_code_str"] = gdf["ZCTA5CE10"].astype(str).str.zfill(5)
    return gdf


@st.cache_resource(show_spinner="🏙️ Loading metro area boundaries...")
def load_cbsa_shapes() -> gpd.GeoDataFrame:
    """加载 CBSA（大都市统计区）边界。"""
    path = _resolve_shapefile_path(CBSA_SHP_PATH, CBSA_ZIP_PATH, "CBSA")
    gdf = gpd.read_file(path)

    if "NAME" not in gdf.columns:
        raise RuntimeError("CBSA shapefile 缺少 'NAME' 列。")

    gdf["name_lower"] = gdf["NAME"].astype(str).str.lower()
    return gdf


# =========================
# 2. City / CBSA 匹配工具
# =========================

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
    # 去重，保持顺序
    return list(dict.fromkeys(tokens))


def resolve_manual_cbsa_name(city: str, city_full: str):
    key = (city_full or city or "").strip().lower()
    if key in MANUAL_CBSA_NAME_MAP:
        return MANUAL_CBSA_NAME_MAP[key]
    # 特例：Boston 一类
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
    根据 city(city_full) 把每个 metro 匹配到一个 CBSA polygon。
    输出一个 GeoDataFrame，用于 metro-level Choropleth。
    """
    cbsa_gdf = _cbsa_gdf.copy()
    if "name_lower" not in cbsa_gdf.columns:
        cbsa_gdf["name_lower"] = cbsa_gdf["NAME"].astype(str).str.lower()

    # 预先算好 CBSA 的质心，方便用 (lat, lon) 选最近的一个
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

        # 1. 手动映射（DC、Boston 等特殊情况）
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

        # 2. 直接用 city_full 做精确匹配 / contains
        city_full_lower = city_full.lower()
        exact = cbsa_gdf[cbsa_name_lower == city_full_lower]
        if exact.empty:
            contains = cbsa_gdf[cbsa_name_lower.str.contains(city_full_lower, na=False)]
        else:
            contains = exact
        candidates = contains

        # 3. 用 city + state token 做模糊匹配
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

        # 多个候选时，用 (lat, lon) 离得最近的
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
    给定 selected_city，返回：
      - zip_df_city: 这个 metro 里、每个 ZIP 的 metric 值
      - gdf_merge: ZCTA polygon + metric merge 后的 GeoDataFrame
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
