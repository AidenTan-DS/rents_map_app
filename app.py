import os
import json
import math

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from streamlit_plotly_events import plotly_events

import geopandas as gpd
from shapely.affinity import translate, scale


DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
DATABRICKS_HTTP_PATH = st.secrets["DATABRICKS_HTTP_PATH"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]


# =========================================================================
# 1. Page config
# =========================================================================

st.set_page_config(
    page_title="Interactive Metro → ZIP Sale Price Map",
    layout="wide"
)

# Hide Plotly modebar
st.markdown(
    """
    <style>
    .modebar {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================================
# 2. Constants & city coords
# =========================================================================

HOUSE_TABLE = "workspace.data511.house_ts"
ZIP_GEO_TABLE = "workspace.data511.zip_geo"

CBSA_JSON_PATH = "data/cbsa.json"
ZCTA_JSON_PATH = "data/zcta.json"


US_CENTER_LAT = 39.8283
US_CENTER_LON = -98.5795
US_ZOOM_LEVEL = 3.8

COORDS = {
    "atlanta, ga": [33.7490, -84.3880],
    "austin, tx": [30.2672, -97.7431],
    "boston, ma": [42.3601, -71.0589],
    "baltimore, md": [39.2904, -76.6122],
    "chicago, il": [41.8781, -87.6298],
    "cincinnati, oh": [39.1031, -84.5120],
    "charlotte, nc": [35.2271, -80.8431],
    "dallas, tx": [32.7767, -96.7970],
    "washington, dc": [38.9072, -77.0369],
    "denver, co": [39.7392, -104.9903],
    "detroit, mi": [42.3314, -83.0458],
    "houston, tx": [29.7604, -95.3698],
    "los angeles, ca": [34.0522, -118.2437],
    "las vegas, nv": [36.1699, -115.1398],
    "miami, fl": [25.7617, -80.1918],
    "minneapolis, mn": [44.9778, -93.2650],
    "new york, ny": [40.7128, -74.0060],
    "orlando, fl": [28.5383, -81.3792],
    "portland, or": [45.5152, -122.6784],
    "pittsburgh, pa": [40.4406, -79.9959],
    "philadelphia, pa": [39.9526, -75.1652],
    "phoenix, az": [33.4484, -112.0740],
    "riverside, ca": [33.9806, -117.3755],
    "san antonio, tx": [29.4241, -98.4936],
    "sacramento, ca": [38.5816, -121.4944],
    "san diego, ca": [32.7157, -117.1611],
    "seattle, wa": [47.6062, -122.3321],
    "san francisco, ca": [37.7749, -122.4194],
    "st. louis, mo": [38.6270, -90.1994],
    "tampa, fl": [27.9506, -82.4572],
}

# Hexbin tile map coordinates (col, row) - manually positioned to approximate geography
HEX_POSITIONS = {
    # West Coast
    "seattle, wa": (1, 1),
    "portland, or": (1, 2),
    "san francisco, ca": (0, 3),
    "sacramento, ca": (1, 3),
    "los angeles, ca": (1, 4),
    "san diego, ca": (1, 5),
    "riverside, ca": (2, 5),
    
    # Mountain/Southwest
    "las vegas, nv": (2, 3),
    "phoenix, az": (3, 5),
    "denver, co": (4, 2),
    
    # Texas
    "dallas, tx": (6, 4),
    "austin, tx": (6, 5),
    "houston, tx": (7, 5),
    "san antonio, tx": (6, 6),
    
    # Midwest
    "minneapolis, mn": (6, 1),
    "chicago, il": (7, 2),
    "detroit, mi": (8, 1),
    "st. louis, mo": (7, 3),
    "cincinnati, oh": (8, 3),
    
    # South
    "atlanta, ga": (9, 4),
    "charlotte, nc": (10, 3),
    "miami, fl": (11, 6),
    "orlando, fl": (11, 5),
    "tampa, fl": (10, 6),
    
    # Northeast
    "pittsburgh, pa": (9, 2),
    "philadelphia, pa": (10, 2),
    "baltimore, md": (10, 3),
    "washington, dc": (11, 3),
    "new york, ny": (11, 1),
    "boston, ma": (12, 1),
}

# =========================================================================
# 3. Helpers: DB & shapes
# =========================================================================

@st.cache_data
def sql_query(query: str) -> pd.DataFrame:
    """Execute SQL query against Databricks warehouse"""
    import databricks.sql as dbsql

    with dbsql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


@st.cache_data
def load_all_data() -> pd.DataFrame:
    """Load all housing data from Databricks and calculate PTI"""
    query = f"""
        SELECT
            h.city,
            h.city_full,
            h.zip_code,
            h.year,
            AVG(h.median_sale_price) AS median_sale_price,
            AVG(h.per_capita_income) AS per_capita_income,
            g.lat,
            g.lon
        FROM {HOUSE_TABLE} h
        JOIN {ZIP_GEO_TABLE} g
          ON CAST(h.zip_code AS STRING) = g.zip_code
        GROUP BY
            h.city, h.city_full, h.zip_code, h.year,
            g.lat, g.lon
    """
    df = sql_query(query)
    df["zip_code_str"] = df["zip_code"].astype(str)
    df["city_clean"] = df["city"].str.lower().str.strip()
    df["year"] = df["year"].astype(int)
    
    # Calculate PTI (Price-to-Income ratio), safely handle division by zero and NaN
    df["pti"] = df.apply(
        lambda row: row["median_sale_price"] / row["per_capita_income"] 
        if pd.notna(row["per_capita_income"]) and pd.notna(row["median_sale_price"]) and row["per_capita_income"] > 0 
        else None, 
        axis=1
    )
    
    return df

@st.cache_resource
def load_zcta_shapes() -> gpd.GeoDataFrame:
    """Load ZIP Code Tabulation Area (ZCTA) shapes from GeoJSON"""
    gdf = gpd.read_file(ZCTA_JSON_PATH)   # ✅ 改成 JSON 变量
    if "ZCTA5CE10" not in gdf.columns:
        raise RuntimeError("ZCTA shapefile missing 'ZCTA5CE10'.")
    gdf["zip_code_str"] = gdf["ZCTA5CE10"].astype(str)
    return gdf

@st.cache_resource
def load_cbsa_shapes() -> gpd.GeoDataFrame:
    """Load Core-Based Statistical Area (CBSA) metro shapes from GeoJSON"""
    gdf = gpd.read_file(CBSA_JSON_PATH)   # ✅ 改成 JSON 变量
    if "NAME" not in gdf.columns:
        raise RuntimeError("CBSA shapefile missing 'NAME'.")
    gdf["name_lower"] = gdf["NAME"].str.lower()
    return gdf


@st.cache_data
def build_city_cbsa_polygons(
    df_city: pd.DataFrame,
    _cbsa_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Match city data with CBSA polygons based on city names"""
    cbsa_gdf = _cbsa_gdf
    records = []

    for _, row in df_city.iterrows():
        city = row["city"]
        city_full = row["city_full"]
        avg_value = row["avg_metric_value"]

        # Extract base city name and find matching CBSA
        base_name = city.split(",")[0].strip().lower()
        matches = cbsa_gdf[cbsa_gdf["name_lower"].str.contains(base_name)]

        if not matches.empty:
            mrow = matches.iloc[0]
            geom = mrow.geometry
            metro_name = mrow["NAME"]

            records.append(
                {
                    "city": city,
                    "city_full": city_full,
                    "metro_name": metro_name,
                    "avg_metric_value": avg_value,
                    "geometry": geom,
                }
            )

    if not records:
        return gpd.GeoDataFrame(
            columns=[
                "city",
                "city_full",
                "metro_name",
                "avg_metric_value",
                "geometry",
            ]
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=cbsa_gdf.crs)

@st.cache_data
def get_zip_polygons_for_metro(selected_city, selected_year, _zcta_shapes, df_filtered):
    """Get ZIP code polygons for a specific metro area"""
    zip_df_city = (
        df_filtered[df_filtered["city"] == selected_city]
        .dropna(subset=["lat", "lon"])
        .reset_index(drop=True)
    )

    if zip_df_city.empty:
        return zip_df_city, gpd.GeoDataFrame()

    zip_df_small = zip_df_city[
        ["zip_code_str", "metric_value", "city_full"]
    ].drop_duplicates()

    gdf_merge = _zcta_shapes.merge(zip_df_small, on="zip_code_str", how="inner")
    return zip_df_city, gdf_merge

# =========================================================================
# 4. Plotting: metro views
# =========================================================================

def create_city_choropleth(
    df_city: pd.DataFrame,
    cbsa_gdf: gpd.GeoDataFrame,
    map_style: str,
    metric_name: str,
):
    """Create choropleth map of metro areas on US basemap"""
    if df_city.empty:
        return None

    # Filter out NaN values
    df_city = df_city[df_city["avg_metric_value"].notna()].copy()
    
    if df_city.empty:
        st.warning(f"No valid data for {metric_name}")
        return None

    city_polygons = build_city_cbsa_polygons(df_city, cbsa_gdf)
    if city_polygons.empty:
        return None

    geojson = json.loads(city_polygons.to_json())

    fig = px.choropleth_mapbox(
        city_polygons,
        geojson=geojson,
        locations="city",
        featureidkey="properties.city",
        color="avg_metric_value",
        mapbox_style=map_style,
        center={"lat": US_CENTER_LAT, "lon": US_CENTER_LON},
        zoom=US_ZOOM_LEVEL,
        height=700,
        color_continuous_scale="RdYlGn_r",
        custom_data=["city", "metro_name", "avg_metric_value"],
    )

    if metric_name == "Price-to-Income Ratio (PTI)":
        hover_template = (
            "<b>%{customdata[1]}</b><br>"
            "Primary city: %{customdata[0]}<br>"
            "Avg PTI: %{customdata[2]:.2f}x<extra></extra>"
        )
    else:
        hover_template = (
            "<b>%{customdata[1]}</b><br>"
            "Primary city: %{customdata[0]}<br>"
            "Avg monthly median sale price: $%{customdata[2]:,.0f}<extra></extra>"
        )

    fig.update_traces(hovertemplate=hover_template)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode="event+select",
        
        coloraxis_colorbar=dict(
            title=metric_name,
            tickprefix="" if "PTI" in metric_name else "$",
            tickformat=",.2f" if "PTI" in metric_name else ",",
            ticksuffix="x" if "PTI" in metric_name else "",
        ),
        hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Arial"
    )
    )
    return fig

def create_packed_metro_polygons_with_lookup(
    df_city: pd.DataFrame,
    cbsa_gdf: gpd.GeoDataFrame,
    metric_name: str,
):
    """Create packed metro visualization with normalized shapes in a grid layout"""
    if df_city.empty:
        return None, []

    # Filter out NaN values
    df_city = df_city[df_city["avg_metric_value"].notna()].copy()
    
    if df_city.empty:
        st.warning(f"No valid data for {metric_name}")
        return None, []

    city_polygons = build_city_cbsa_polygons(df_city, cbsa_gdf)
    if city_polygons.empty:
        return None, []

    coords = df_city[["city", "lat", "lon"]].drop_duplicates()
    gdf = city_polygons.merge(coords, on="city", how="left")
    gdf = gdf.to_crs(epsg=2163)

    # Normalize all shapes to similar size
    target_size = 1.2
    records = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        max_dim = max(maxx - minx, maxy - miny)
        if max_dim == 0:
            continue

        factor = target_size / max_dim
        geom_scaled = scale(geom, xfact=factor, yfact=factor, origin="center")
        c = geom_scaled.centroid
        geom_norm = translate(geom_scaled, xoff=-c.x, yoff=-c.y)

        records.append(
            {
                "city": row["city"],
                "city_full": row["city_full"],
                "metro_name": row["metro_name"],
                "avg_metric_value": row["avg_metric_value"],
                "lat": row["lat"],
                "lon": row["lon"],
                "geometry": geom_norm,
            }
        )

    if not records:
        return None, []

    gdf_norm = gpd.GeoDataFrame(records).reset_index(drop=True)

    # Calculate grid dimensions
    n_cities = len(gdf_norm)
    area_per_city = 1.8
    grid_width = int(math.sqrt(n_cities * 1.6 * area_per_city)) + 2
    grid_height = int(math.sqrt(n_cities / 1.6 * area_per_city)) + 2

    min_lon, max_lon = gdf_norm["lon"].min(), gdf_norm["lon"].max()
    min_lat, max_lat = gdf_norm["lat"].min(), gdf_norm["lat"].max()
    lon_span = (max_lon - min_lon) or 1
    lat_span = (max_lat - min_lat) or 1

    # Pack shapes into grid layout based on geographic position
    occupied_cells = {}
    gdf_norm["grid_x"] = 0.0
    gdf_norm["grid_y"] = 0.0
    gdf_norm = gdf_norm.sort_values("lon").reset_index(drop=True)

    for idx, row in gdf_norm.iterrows():
        norm_x = (row["lon"] - min_lon) / lon_span
        norm_y = (row["lat"] - min_lat) / lat_span
        target_x = int(round(norm_x * (grid_width - 1)))
        target_y = int(round(norm_y * (grid_height - 1)))

        # Find nearest available cell
        best_x, best_y = target_x, target_y
        found = False
        radius = 0
        while not found:
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        candidates.append((target_x + dx, target_y + dy))
            candidates.sort(
                key=lambda p: (p[0] - target_x) ** 2 + (p[1] - target_y) ** 2
            )
            for cx, cy in candidates:
                if (cx, cy) not in occupied_cells:
                    best_x, best_y = cx, cy
                    found = True
                    break
            radius += 1
            if radius > 25:
                break

        occupied_cells[(best_x, best_y)] = row["city"]
        gdf_norm.at[idx, "grid_x"] = best_x * 1.5
        gdf_norm.at[idx, "grid_y"] = best_y * 1.8

    # Create plotly figure with colored polygons
    fig = go.Figure()
    trace_lookup = []
    annotations = []

    vmin = float(gdf_norm["avg_metric_value"].min())
    vmax = float(gdf_norm["avg_metric_value"].max())
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    
    # Ensure no NaN in normalization
    norm_vals = (gdf_norm["avg_metric_value"] - vmin) / (vmax - vmin)
    norm_vals = norm_vals.fillna(0.5)
    
    colors = sample_colorscale("Viridis", norm_vals.tolist())

    for idx, row in gdf_norm.iterrows():
        gx = row["grid_x"]
        gy = row["grid_y"]
        base_geom = row.geometry
        geoms = (
            list(base_geom.geoms)
            if base_geom.geom_type == "MultiPolygon"
            else [base_geom]
        )

        metric_val = row["avg_metric_value"]
        metro_name = row["metro_name"]
        primary_city = row["city"]

        state_abbr = ""
        if "," in primary_city:
            state_abbr = primary_city.split(",")[1].strip()

        # Format hover text and labels based on metric type
        if metric_name == "Price-to-Income Ratio (PTI)":
            hover_text = (
                f"<b>{metro_name}</b><br>"
                f"Primary city: {primary_city}<br>"
                f"Avg PTI: {metric_val:.2f}x"
            )
            label_text = f"<b>{state_abbr}</b><br>{metric_val:.1f}x"
        else:
            hover_text = (
                f"<b>{metro_name}</b><br>"
                f"Primary city: {primary_city}<br>"
                f"Avg monthly median sale price: ${metric_val:,.0f}"
            )
            price_k = int(metric_val / 1000)
            label_text = f"<b>{state_abbr}</b><br>${price_k}K"

        color = colors[idx]
        custom_row = [primary_city, metro_name, metric_val]

        # Add polygon traces
        for g in geoms:
            geom_shifted = translate(g, xoff=gx, yoff=gy)
            x, y = geom_shifted.exterior.coords.xy

            fig.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(y),
                    mode="lines",
                    fill="toself",
                    fillcolor=color,
                    line=dict(width=1, color="rgba(255,255,255,0.9)"),
                    name=metro_name,
                    showlegend=False,
                    hoverinfo="text",
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    customdata=[custom_row] * len(x),
                )
            )
            trace_lookup.append(primary_city)

        # Add text annotations
        annotations.append(
            dict(
                x=gx,
                y=gy - 0.85,
                text=label_text,
                showarrow=False,
                font=dict(size=10, family="Arial", color="#333"),
                captureevents=False,
                bgcolor="rgba(255,255,255,0.75)",
                borderpad=3,
                borderwidth=0,
            )
        )

    # Add invisible colorbar trace
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale="Viridis",
                cmin=vmin,
                cmax=vmax,
                color=[vmin, vmax],
                showscale=True,
                colorbar=dict(
                    title=metric_name,
                    tickprefix="" if "PTI" in metric_name else "$",
                    tickformat=",.2f" if "PTI" in metric_name else ",",
                    ticksuffix="x" if "PTI" in metric_name else "",
                    orientation="v",
                    titleside="right",
                    x=1.02,
                    thickness=14,
                    len=0.6,
                ),
            ),
            showlegend=False,
            hoverinfo="none",
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=750,
        plot_bgcolor="#f4f4f4",
        paper_bgcolor="white",
        dragmode="pan",
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        ),
        annotations=annotations,
        title=dict(
            text="💡 Hover for details • Click to explore ZIPs • 🎨 " + 
                 ("Darker colors = Higher PTI (Less Affordable)" if "PTI" in metric_name else "Darker colors = Higher Prices"),
            x=0.5,
            y=0.99,
            xanchor='center',
            yanchor='top',
            font=dict(size=13, color="#666", family="Arial")
        )
    )

    return fig, trace_lookup

def create_hexbin_tile_map(
    df_city: pd.DataFrame,
    metric_name: str,
):
    """Create hexagonal tile map with metros positioned geographically like US state hex maps"""
    if df_city.empty:
        return None, []

    # Filter out NaN values
    df_city = df_city[df_city["avg_metric_value"].notna()].copy()
    
    if df_city.empty:
        st.warning(f"No valid data for {metric_name}")
        return None, []

    # Hexagon geometry parameters
    hex_size = 1.0
    hex_height = hex_size * math.sqrt(3)
    x_spacing = hex_size * 1.5
    y_spacing = hex_height
    
    # Offset to shift entire map left
    X_SHIFT = -2.0  # Shift left by 2 units
    
    # Helper function to create hexagon vertices
    def create_hexagon(cx, cy, size):
        """Create hexagon vertices centered at (cx, cy)"""
        angles = np.linspace(0, 2 * np.pi, 7)
        x = cx + size * np.cos(angles)
        y = cy + size * np.sin(angles)
        return x, y
    
    fig = go.Figure()
    trace_lookup = []
    annotations = []
    
    # Color scale
    vmin = float(df_city["avg_metric_value"].min())
    vmax = float(df_city["avg_metric_value"].max())
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    
    # Create color mapping
    norm_vals = (df_city["avg_metric_value"] - vmin) / (vmax - vmin)
    norm_vals = norm_vals.fillna(0.5)
    colors = sample_colorscale("Viridis", norm_vals.tolist())
    
    # Create lookup dict using city_clean (already normalized)
    city_lookup = {}
    for idx, row in df_city.iterrows():
        city_lookup[row["city_clean"]] = {
            "metric_val": row["avg_metric_value"],
            "city_full": row["city_full"],
            "city": row["city"],
            "color": colors[idx]
        }
    
    # Place hexagons according to predefined positions
    hexagons_placed = 0
    for city_clean, (col, row) in HEX_POSITIONS.items():
        # Check if city exists in data using city_clean
        if city_clean not in city_lookup:
            continue
        
        city_info = city_lookup[city_clean]
        metric_val = city_info["metric_val"]
        city_full = city_info["city_full"]
        city = city_info["city"]
        color = city_info["color"]
        
        # Calculate hexagon center with offset for odd rows, then shift left
        x_offset = (col * x_spacing) + (0.75 * hex_size if row % 2 == 1 else 0) + X_SHIFT
        y_offset = -row * y_spacing
        
        # Extract state abbreviation
        state_abbr = ""
        if "," in city:
            state_abbr = city.split(",")[1].strip().upper()
        
        # Create hexagon
        hex_x, hex_y = create_hexagon(x_offset, y_offset, hex_size * 0.95)
        
        # Format hover text and label
        if metric_name == "Price-to-Income Ratio (PTI)":
            hover_text = (
                f"<b style='font-size:16px'>{city_full}</b><br>"
                f"<span style='font-size:15px'>Primary city: {city}</span><br>"
                f"<span style='font-size:15px'>Avg PTI: {metric_val:.2f}x</span>"
            )
            label_text = f"<b>{state_abbr}</b><br>{metric_val:.1f}x"
        else:
            hover_text = (
                f"<b style='font-size:16px'>{city_full}</b><br>"
                f"<span style='font-size:15px'>Primary city: {city}</span><br>"
                f"<span style='font-size:15px'>Avg monthly median sale price: ${metric_val:,.0f}</span>"
            )
            price_k = int(metric_val / 1000)
            label_text = f"<b>{state_abbr}</b><br>${price_k}K"
        
        custom_row = [city, city_full, metric_val]
        
        # Add hexagon trace
        fig.add_trace(
            go.Scatter(
                x=hex_x,
                y=hex_y,
                mode="lines",
                fill="toself",
                fillcolor=color,
                line=dict(width=2, color="white"),
                name=city_full,
                showlegend=False,
                hoverinfo="text",
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                customdata=[custom_row] * len(hex_x),
                hoverlabel=dict(
                    bgcolor="white",
                    font=dict(size=15, family="Arial", color="#333"),
                    bordercolor="#ccc"
                )
            )
        )
        trace_lookup.append(city)
        
        # Add text annotation
        annotations.append(
            dict(
                x=x_offset,
                y=y_offset - 0.15,
                text=label_text,
                showarrow=False,
                font=dict(size=9, family="Arial", color="#333"),
                captureevents=False,
                bgcolor="rgba(255,255,255,0.7)",
                borderpad=2,
                borderwidth=0,
            )
        )
        
        hexagons_placed += 1
    
    if hexagons_placed == 0:
        st.warning("⚠️ No hexagons could be placed. City names may not match.")
        return None, []
    
    # Add invisible colorbar trace
    # Add invisible colorbar trace (simplified for compatibility)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale="Viridis",
                cmin=vmin,
                cmax=vmax,
                color=[vmin, vmax],
                showscale=True,   # 只开显示，不自定义 colorbar 细节
            ),
            showlegend=False,
            hoverinfo="none",
        )
    )

    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=750,
        plot_bgcolor="#f0f0f0",
        paper_bgcolor="white",
        dragmode="pan",
        annotations=annotations,
    )
    
    return fig, trace_lookup

# =========================================================================
# 5. Plotting: ZIP view
# =========================================================================

def create_zip_choropleth(gdf, map_style, city_coords, center_df, metric_name):
    """Create choropleth map of ZIP codes for a specific metro"""
    if gdf.empty:
        return None

    # Filter out NaN values
    gdf = gdf[gdf["metric_value"].notna()].copy()
    
    if gdf.empty:
        st.warning(f"No valid data for {metric_name}")
        return None

    geojson = json.loads(gdf.to_json())

    # Determine map center
    if city_coords:
        center_lat, center_lon = city_coords
    elif center_df is not None and not center_df.empty:
        center_lat = center_df["lat"].mean()
        center_lon = center_df["lon"].mean()
    else:
        center_lat = gdf.geometry.centroid.y.mean()
        center_lon = gdf.geometry.centroid.x.mean()

    fig = px.choropleth_mapbox(
        gdf,
        geojson=geojson,
        locations="zip_code_str",
        featureidkey="properties.zip_code_str",
        color="metric_value",
        mapbox_style=map_style,
        center={"lat": center_lat, "lon": center_lon},
        zoom=9,
        height=700,
        color_continuous_scale="Viridis",
        custom_data=["zip_code_str", "city_full"],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>ZIP %{customdata[0]}</b><br>"
            "Metro: %{customdata[1]}<extra></extra>"
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode="event+select",
        coloraxis_colorbar=dict(
            title=metric_name,
            tickprefix="" if "PTI" in metric_name else "$",
            tickformat=",.2f" if "PTI" in metric_name else ",",
            ticksuffix="x" if "PTI" in metric_name else "",
        ),
        hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Arial"
    )
    )
    return fig

# =========================================================================
# 6. Session state initialization
# =========================================================================

if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "city"
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = None
if "selected_zip" not in st.session_state:
    st.session_state["selected_zip"] = None

# =========================================================================
# 7. Load data from Databricks
# =========================================================================

try:
    df_all = load_all_data()
except Exception as e:
    st.error(f"❌ Failed to read Databricks tables: {e}")
    st.stop()

if df_all.empty:
    st.warning("⚠️ No data loaded from database.")
    st.stop()

# =========================================================================
# 8. Sidebar controls
# =========================================================================

with st.sidebar:
    st.title("⚙️ Controls")
    
    st.markdown("### 📅 Time Period")
    min_year = int(df_all["year"].min())
    max_year = int(df_all["year"].max())
    selected_year = st.slider("Select year", min_year, max_year, max_year)
    st.caption(f"📊 Data range: {min_year} - {max_year}")

    st.markdown("### 📊 Metric")
    metric_type = st.radio(
        "Choose what to display",
        ["Median Sale Price", "Price-to-Income Ratio (PTI)"],
        index=0,
        help="PTI shows affordability: lower ratio = more affordable"
    )

    st.markdown("### 🗺️ Metro View Style")
    metro_view_type = st.radio(
        "Metro-level visualization",
        ["Hexbin Tile Map", "Packed metro shapes (no map)", "US basemap (CBSA polygons)"],
        index=0,
    )
    
    # Only show basemap style selector when using basemap view
    if metro_view_type == "US basemap (CBSA polygons)":
        map_style = st.selectbox(
            "Basemap style",
            ["carto-positron", "open-street-map", "carto-darkmatter"],
            index=0,
        )
    else:
        map_style = "carto-positron"  # Default for ZIP view

    # Quick metro search (only shown in metro view)
    if st.session_state["view_mode"] == "city":
        st.markdown("---")
        st.markdown("### 🔍 Quick Metro Search")
        st.caption("Jump directly to a metro area")

        df_filtered_sidebar = df_all[df_all["year"] == selected_year].copy()
        if not df_filtered_sidebar.empty:
            df_city_sidebar = (
                df_filtered_sidebar.groupby(["city", "city_full"], as_index=False)
                .agg(avg_median_sale_price=("median_sale_price", "mean"))
            )
            metro_list = df_city_sidebar.sort_values('city')['city'].tolist()

            selected_from_sidebar = st.selectbox(
                "Type to search:",
                [""] + metro_list,
                format_func=lambda x: "🔎 Start typing metro name..." if x == "" else f"📍 {x}"
            )

            if selected_from_sidebar and st.button("➡️ View ZIP codes"):
                st.session_state["selected_city"] = selected_from_sidebar
                st.session_state["view_mode"] = "zip"
                st.session_state["selected_zip"] = None
                st.rerun()

# =========================================================================
# 9. Filter data by year and set metric column
# =========================================================================

df_filtered = df_all[df_all["year"] == selected_year].copy()
if df_filtered.empty:
    st.warning(f"""
    ### ⚠️ No data available for {selected_year}
    
    **Try this:**
    - Select a different year from the sidebar
    - Available years: {min_year} - {max_year}
    """)
    st.stop()

# Set metric column based on selected metric type and filter out NaN
if metric_type == "Price-to-Income Ratio (PTI)":
    df_filtered = df_filtered[df_filtered["pti"].notna()].copy()
    if df_filtered.empty:
        st.warning(f"⚠️ No valid PTI data for {selected_year}.")
        st.stop()
    df_filtered["metric_value"] = df_filtered["pti"]
    metric_column = "pti"
else:
    df_filtered = df_filtered[df_filtered["median_sale_price"].notna()].copy()
    if df_filtered.empty:
        st.warning(f"⚠️ No valid price data for {selected_year}.")
        st.stop()
    df_filtered["metric_value"] = df_filtered["median_sale_price"]
    metric_column = "median_sale_price"

# Aggregate by city
df_city = (
    df_filtered.groupby(["city", "city_full", "city_clean"], as_index=False)
    .agg(
        n=("zip_code_str", "count"),
        avg_metric_value=(metric_column, "mean"),
    )
)
df_city["lat"] = df_city["city_clean"].apply(lambda x: COORDS.get(x, [None, None])[0])
df_city["lon"] = df_city["city_clean"].apply(lambda x: COORDS.get(x, [None, None])[1])
df_city_map = df_city.dropna(subset=["lat", "lon"]).reset_index(drop=True)

# =========================================================================
# 10. Header and usage instructions
# =========================================================================

st.title("🏙️ Metro → ZIP Sale Price Explorer")

# App introduction (collapsible)
with st.expander("ℹ️ How to use this app", expanded=False):
    st.markdown("""
    **Welcome!** This interactive tool helps you explore housing affordability across US metro areas.
    
    **How it works:**
    1. **Select a year** and metric from the sidebar (left panel)
    2. **Metro View**: Hover over metros to see details, click any metro to drill down
    3. **ZIP View**: Click any ZIP code to see its historical trends
    4. Use the "⬅️ Back to All Metros" button in ZIP view to return
    
    **Metrics explained:**
    - **Median Sale Price**: Average monthly median home sale price in dollars
    - **PTI (Price-to-Income Ratio)**: Home price ÷ per capita income  
      → Lower PTI = More affordable | Higher PTI = Less affordable
    """)

current_metro_name = None
if st.session_state["selected_city"]:
    row = df_city_map[df_city_map["city"] == st.session_state["selected_city"]]
    if not row.empty:
        current_metro_name = row["city_full"].iloc[0]

st.markdown("---")

# =========================================================================
# 11. Main view: metro or ZIP
# =========================================================================

if st.session_state["view_mode"] == "city":
    # =========================================================================
    # METRO VIEW
    # =========================================================================
    
    # Display instructions based on view type
    col_hdr1, col_hdr2 = st.columns([4, 1])
    with col_hdr1:
        if metro_view_type == "Hexbin Tile Map":
            st.info(
                f"📐 **Hexbin Tile Map ({selected_year})** — Showing **{metric_type}**\n\n"
                f"💡 **How to interact:** Hover over any hexagon for details | Click to explore ZIP codes\n\n"
                f"🎨 **Layout:** Geographic positioning similar to US state tile maps | Darker colors = {'Higher PTI (Less Affordable)' if 'PTI' in metric_type else 'Higher Prices'}"
            )
        elif metro_view_type == "Packed metro shapes (no map)":
            st.info(
                f"📍 **Interactive Metro View ({selected_year})** — Showing **{metric_type}**\n\n"
                f"💡 **How to interact:** Hover over any metro shape for details | Click to explore ZIP codes\n\n"
                f"🎨 **Color guide:** {'Darker = Higher PTI (Less Affordable)' if 'PTI' in metric_type else 'Darker = Higher Prices'}"
            )
        else:
            st.info(
                f"📍 **Metro View ({selected_year})** — Showing **{metric_type}**\n\n"
                f"💡 **How to interact:** Hover for details | Click any metro to drill down to ZIP-level"
            )

    fig_city = None
    trace_lookup_map = []

    # Create appropriate visualization based on selection
    if metro_view_type == "Hexbin Tile Map":
        fig_city, trace_lookup_map = create_hexbin_tile_map(df_city_map, metric_type)
    elif metro_view_type == "Packed metro shapes (no map)":
        # Load CBSA shapefiles
        try:
            cbsa_shapes = load_cbsa_shapes()
            fig_city, trace_lookup_map = create_packed_metro_polygons_with_lookup(
                df_city_map, cbsa_shapes, metric_type
            )
        except Exception as e:
            st.error(f"❌ Shapefile Error: {e}")
    else:  # US basemap (CBSA polygons)
        try:
            cbsa_shapes = load_cbsa_shapes()
            fig_city = create_city_choropleth(df_city_map, cbsa_shapes, map_style, metric_type)
        except Exception as e:
            st.error(f"❌ Shapefile Error: {e}")

    # Display map and handle click events
    if fig_city is not None:
        selected_points = plotly_events(
            fig_city,
            click_event=True,
            select_event=False,
            key=f"metro_click_{metro_view_type}",
            override_height=750,
        )

        if selected_points:
            pt = selected_points[0]
            clicked_city = None

            # Try to extract city from customdata
            custom = pt.get("customdata")
            if custom:
                first = custom[0] if isinstance(custom[0], (list, tuple)) else custom
                if isinstance(first, (list, tuple)) and len(first) > 0:
                    clicked_city = first[0]
                else:
                    clicked_city = first

            # Fallback for hexbin/packed view: use trace lookup
            if not clicked_city and metro_view_type in ["Hexbin Tile Map", "Packed metro shapes (no map)"]:
                curve_num = pt.get("curveNumber")
                if curve_num is not None and 0 <= curve_num < len(trace_lookup_map):
                    clicked_city = trace_lookup_map[curve_num]

            # Fallback for basemap view: use location or pointIndex
            if not clicked_city and metro_view_type == "US basemap (CBSA polygons)":
                loc = pt.get("location")
                if loc:
                    clicked_city = loc
                else:
                    idx = pt.get("pointIndex", pt.get("pointNumber"))
                    if idx is not None and 0 <= idx < len(df_city_map):
                        clicked_city = df_city_map.iloc[idx]["city"]

            # Transition to ZIP view if city clicked
            if clicked_city and clicked_city != st.session_state["selected_city"]:
                st.session_state["selected_city"] = clicked_city
                st.session_state["selected_zip"] = None
                st.session_state["view_mode"] = "zip"
                st.rerun()

else:
    # =========================================================================
    # ZIP VIEW
    # =========================================================================
    
    selected_city = st.session_state["selected_city"]
    if not selected_city:
        st.warning("⚠️ No metro selected. Please use the sidebar search to select a metro.")
        st.stop()

    # Breadcrumb navigation
    st.markdown(f"### 🗺️ Navigation: `USA` → `{current_metro_name or selected_city}` → `ZIP Codes`")
    
    col_back, col_space = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Back to All Metros"):
            st.session_state["view_mode"] = "city"
            st.session_state["selected_city"] = None
            st.session_state["selected_zip"] = None
            st.rerun()
    
    st.markdown("---")
    
    # Display instructions
    label = current_metro_name or selected_city
    st.info(
        f"📍 **ZIP Code View: {label}** ({selected_year}) — Showing **{metric_type}**\n\n"
        f"💡 **How to interact:** Click any ZIP code on the map to see detailed history and comparison"
    )

    # Load ZIP shapes and filter data
    try:
        zcta_shapes = load_zcta_shapes()
        zip_df_city, gdf_merge = get_zip_polygons_for_metro(
            selected_city, selected_year, zcta_shapes, df_filtered
        )
    except Exception as e:
        st.error(f"❌ ZIP Shapefile Error: {e}")
        zip_df_city, gdf_merge = pd.DataFrame(), gpd.GeoDataFrame()

    if gdf_merge.empty or zip_df_city.empty:
        st.warning(f"""
        ### ⚠️ No ZIP code data available
        
        **Metro:** {selected_city}  
        **Year:** {selected_year}
        
        **Try this:**
        - Select a different year from the sidebar
        - Or go back and choose another metro
        """)
    else:
        # Filter out NaN values
        zip_df_city = zip_df_city[zip_df_city["metric_value"].notna()].copy()
        
        if zip_df_city.empty:
            st.warning(f"⚠️ No valid {metric_type} data for {selected_city} in {selected_year}.")
        else:
            # Auto-select first ZIP if none selected
            if st.session_state.get("selected_zip") is None and not zip_df_city.empty:
                st.session_state["selected_zip"] = zip_df_city["zip_code_str"].iloc[0]
                st.rerun()

            # Create two-column layout: map and details
            col_map, col_detail = st.columns([2.2, 1])

            with col_map:
                # Create ZIP choropleth map
                city_coords = COORDS.get(selected_city.lower().strip())
                fig_zip = create_zip_choropleth(
                    gdf_merge, map_style, city_coords, zip_df_city, metric_type
                )

                if fig_zip is not None:
                    selected_zip_pts = plotly_events(
                        fig_zip,
                        click_event=True,
                        select_event=False,
                        key=f"zip_click_{selected_city}_{selected_year}",
                        override_height=700,
                    )
                else:
                    selected_zip_pts = []

                # Handle ZIP click events
                if selected_zip_pts:
                    zp = selected_zip_pts[0]
                    clicked_zip = None

                    # Try to extract ZIP from customdata
                    custom = zp.get("customdata")
                    if custom:
                        cd = custom[0] if isinstance(custom[0], (list, tuple)) else custom
                        clicked_zip = str(cd[0])
                    else:
                        idx = zp.get("pointIndex", zp.get("pointNumber"))
                        if idx is not None and 0 <= idx < len(gdf_merge):
                            clicked_zip = str(gdf_merge.iloc[idx]["zip_code_str"])

                    if clicked_zip and clicked_zip != st.session_state.get("selected_zip"):
                        st.session_state["selected_zip"] = clicked_zip
                        st.rerun()

            with col_detail:
                st.subheader("Selected ZIP Details")
                
                # Data export button
                if not zip_df_city.empty:
                    csv = zip_df_city[['zip_code_str', 'year', 'metric_value', 'city_full']].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Data (CSV)",
                        data=csv,
                        file_name=f"{selected_city.replace(',', '_')}_{selected_year}_zipdata.csv",
                        mime="text/csv",
                        help="Download all ZIP code data for this metro"
                    )

                active_zip = st.session_state.get("selected_zip")
                if not active_zip:
                    st.info("👆 Click any ZIP code on the map to see its details")
                else:
                    row_now = zip_df_city[zip_df_city["zip_code_str"] == active_zip]
                    if row_now.empty:
                        st.warning(f"⚠️ No data for ZIP {active_zip} in {selected_year}.")
                    else:
                        metric_val = float(row_now["metric_value"].iloc[0])
                        metro_avg_now = float(zip_df_city["metric_value"].mean())
                        diff = metric_val - metro_avg_now
                        pct_diff = diff / metro_avg_now * 100 if metro_avg_now != 0 else 0.0

                        # Determine affordability label
                        if pct_diff > 10:
                            affordability_label = "above"
                        elif pct_diff < -10:
                            affordability_label = "below"
                        else:
                            affordability_label = "close to"

                        metro_name = row_now["city_full"].iloc[0]

                        # Display current year metrics
                        st.markdown(
                            f"### ZIP `{active_zip}` — **{metro_name} metro** ({selected_year})"
                        )
                        
                        if metric_type == "Price-to-Income Ratio (PTI)":
                            st.markdown(f"**PTI:** `{metric_val:.2f}x`")
                            st.markdown(
                                f"In {selected_year}, this ZIP's PTI is **{affordability_label}** "
                                f"the metro-wide average by about `{pct_diff:+.1f}%`."
                            )
                        else:
                            st.markdown(f"**Avg monthly median sale price:** `${metric_val:,.0f}`")
                            st.markdown(
                                f"In {selected_year}, this ZIP is **{affordability_label}** "
                                f"the metro-wide average by about `{pct_diff:+.1f}%`."
                            )

                        # Historical trend chart
                        st.markdown(f"#### {metric_type} history (by year)")
                        zip_history = (
                            df_all[
                                (df_all["city"] == selected_city)
                                & (df_all["zip_code_str"] == active_zip)
                                & (df_all[metric_column].notna())
                            ]
                            .sort_values("year")
                        )

                        if not zip_history.empty:
                            history_df = (
                                zip_history[["year", metric_column]]
                                .dropna()
                                .sort_values("year")
                                .set_index("year")
                            )
                            history_df = history_df.rename(
                                columns={metric_column: metric_type}
                            )
                            st.line_chart(history_df[metric_type])
                        else:
                            st.info("No historical data available for this ZIP.")

            # Metro-wide summary statistics
            st.markdown("---")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

            values = zip_df_city["metric_value"]
            nonzero_values = values[values > 0]

            with col_m1:
                st.metric("ZIP codes in metro", len(zip_df_city))

            with col_m2:
                if metric_type == "Price-to-Income Ratio (PTI)":
                    st.metric("Metro avg PTI", f"{values.mean():.2f}x")
                else:
                    st.metric("Metro avg price", f"${values.mean():,.0f}")

            with col_m3:
                if metric_type == "Price-to-Income Ratio (PTI)":
                    st.metric("Max PTI", f"{nonzero_values.max():.2f}x" if not nonzero_values.empty else "N/A")
                else:
                    st.metric("Max price", f"${nonzero_values.max():,.0f}" if not nonzero_values.empty else "N/A")

            with col_m4:
                if metric_type == "Price-to-Income Ratio (PTI)":
                    st.metric("Min PTI (non-zero)", f"{nonzero_values.min():.2f}x" if not nonzero_values.empty else "N/A")
                else:
                    st.metric("Min price (non-zero)", f"${nonzero_values.min():,.0f}" if not nonzero_values.empty else "N/A")