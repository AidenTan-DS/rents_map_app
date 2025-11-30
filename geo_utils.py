import os
import zipfile
import tempfile
from pathlib import Path

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


# ============================================================
# 1. 通用：从本地或 GitHub Release ZIP 加载 shapefile
# ============================================================

def _get_zip_url_from_secrets(key: str) -> str:
    """
    从 st.secrets 或环境变量中拿 ZIP 下载链接。
    例如在 .streamlit/secrets.toml 里配置:
        CBSA_ZIP_URL = "https://github.com/.../cbsa_shapes.zip"
        ZCTA_ZIP_URL = "https://github.com/.../zcta_shapes.zip"
    """
    # st.secrets 里优先
    if key in st.secrets:
        return st.secrets[key]
    # 退一步用环境变量
    return os.getenv(key, "")


def _download_and_extract_zip(zip_url: str, label: str) -> Path:
    """
    下载 zip 到临时目录并解压，返回解压后的目录路径。
    这个函数只在 cache 里调用，所以只会执行一次。
    """
    if not zip_url:
        raise RuntimeError(f"{label}: 未配置 ZIP 下载链接（在 secrets.toml 里设置 {label}_ZIP_URL）")

    # 使用 Streamlit 的临时目录
    tmp_root = Path(tempfile.gettempdir()) / "rents_map_shapes"
    tmp_root.mkdir(parents=True, exist_ok=True)

    zip_path = tmp_root / f"{label.lower()}.zip"
    extract_dir = tmp_root / label.lower()
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 如果 zip 已经存在就不再下载（简单一点）
    if not zip_path.exists():
        # 不额外依赖 requests，直接用 urllib
        import urllib.request

        try:
            st.write(f"⬇️ Downloading {label} shapefile ZIP ...")
            urllib.request.urlretrieve(zip_url, zip_path.as_posix())
        except Exception as e:
            raise RuntimeError(f"{label}: 下载 ZIP 失败，请检查 URL 是否正确: {e}")

    # 解压（如果已经解压过，再解压一次也没关系）
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        raise RuntimeError(f"{label}: 解压 ZIP 失败，请确认上传的文件是有效的 zip: {e}")

    return extract_dir


def _find_shp_file(root_dir: Path) -> Path:
    """
    在解压后的目录里递归寻找第一个 .shp 文件。
    假设每个 ZIP 里只放一套 shapefile。
    """
    shp_files = list(root_dir.rglob("*.shp"))
    if not shp_files:
        raise RuntimeError(f"在 {root_dir} 下找不到任何 .shp 文件，请检查 ZIP 内容。")
    return shp_files[0]


def _load_shapefile(local_path: str, zip_url_key: str, label: str) -> gpd.GeoDataFrame:
    """
    优先使用本地 shapefile（例如 data/*.shp），
    本地没有时，从 GitHub Release 的 ZIP 下载并加载。
    """
    local_path_obj = Path(local_path)

    # 1) 本地路径存在：直接读
    if local_path_obj.exists():
        return gpd.read_file(local_path_obj.as_posix())

    # 2) 本地不存在：从 ZIP 下载
    zip_url = _get_zip_url_from_secrets(zip_url_key)
    extract_dir = _download_and_extract_zip(zip_url, label)
    shp_path = _find_shp_file(extract_dir)
    return gpd.read_file(shp_path.as_posix())


# ============================================================
# 2. 加载 ZCTA / CBSA 边界
# ============================================================

@st.cache_resource(show_spinner="🗺️ Loading ZIP code boundaries...")
def load_zcta_shapes() -> gpd.GeoDataFrame:
    """
    加载 ZIP (ZCTA) shapefile：
    - 本地有 ZCTA_SHP_PATH 就用本地
    - 否则从 ZCTA_ZIP_URL 下载 ZIP，解压后自动找 .shp
    """
    gdf = _load_shapefile(ZCTA_SHP_PATH, "ZCTA_ZIP_URL", "ZCTA")

    if "ZCTA5CE10" not in gdf.columns:
        raise RuntimeError("ZCTA shapefile 缺少字段 'ZCTA5CE10'。请确认用的是 Census ZCTA shapefile。")

    gdf["zip_code_str"] = gdf["ZCTA5CE10"].astype(str)
    return gdf


@st.cache_resource(show_spinner="🏙️ Loading metro area boundaries...")
def load_cbsa_shapes() -> gpd.GeoDataFrame:
    """
    加载 CBSA shapefile：
    - 本地有 CBSA_SHP_PATH 就用本地
    - 否则从 CBSA_ZIP_URL 下载 ZIP，解压后自动找 .shp
    """
    gdf = _load_shapefile(CBSA_SHP_PATH, "CBSA_ZIP_URL", "CBSA")

    if "NAME" not in gdf.columns:
        raise RuntimeError("CBSA shapefile 缺少字段 'NAME'。请确认用的是 CBSA shapefile。")

    gdf["name_lower"] = gdf["NAME"].astype(str).str.lower()
    return gdf


# ============================================================
# 3. City / CBSA 匹配逻辑（基本不变）
# ============================================================

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
