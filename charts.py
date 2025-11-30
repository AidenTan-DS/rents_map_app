# charts.py
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import geopandas as gpd
import streamlit as st

from config_data import (
    US_CENTER_LAT,
    US_CENTER_LON,
    US_ZOOM_LEVEL,
    US_BOUNDS,
)
from config_data import get_colorscale
from config_data import compute_rankings
from geo_utils import build_city_cbsa_polygons

# ----------------- METRO LEVEL -----------------
def create_city_choropleth(df_city, cbsa_gdf, map_style, metric_name, is_dark_mode=False):
    if df_city.empty:
        return None, None

    df_city = df_city[df_city["avg_metric_value"].notna()].copy()
    if df_city.empty:
        st.warning(f"No valid data for {metric_name}")
        return None, None

    city_polygons = build_city_cbsa_polygons(df_city, cbsa_gdf, metric_name)
    if city_polygons.empty:
        return None, None

    city_polygons = city_polygons.reset_index(drop=True)
    city_polygons["id"] = city_polygons.index.astype(str)

    city_polygons_4326 = city_polygons.to_crs(epsg=4326)
    city_polygons_proj = city_polygons_4326.to_crs(epsg=2163)
    centroids_proj = city_polygons_proj.geometry.centroid
    centroids_4326 = gpd.GeoSeries(centroids_proj, crs=2163).to_crs(epsg=4326)
    city_polygons_4326["center_lat"] = centroids_4326.y
    city_polygons_4326["center_lon"] = centroids_4326.x

    geojson = json.loads(city_polygons_4326.to_json())
    vmin = float(city_polygons["avg_metric_value"].min())
    vmax = float(city_polygons["avg_metric_value"].max())
    colorscale = get_colorscale(metric_name, is_dark_mode)

    fig = go.Figure()

    hover_texts = []
    for _, row in city_polygons_4326.iterrows():
        rank_text = f"#{int(row['rank'])} of {int(row['rank_total'])}"
        if "PTI" in metric_name:
            hover_texts.append(
                f"<b>{row['metro_name']}</b><br>"
                f"Primary city: {row['city']}<br>"
                f"Avg PTI: {row['avg_metric_value']:.2f}x<br>"
                f"{rank_text}"
            )
        else:
            hover_texts.append(
                f"<b>{row['metro_name']}</b><br>"
                f"Primary city: {row['city']}<br>"
                f"Avg Price: ${row['avg_metric_value']:,.0f}<br>"
                f"{rank_text}"
            )

    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=city_polygons_4326["id"],
            z=city_polygons_4326["avg_metric_value"],
            featureidkey="properties.id",
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
            marker_opacity=0.88,
            marker_line_width=0.8,
            marker_line_color="rgba(249,250,251,0.8)"
            if not is_dark_mode
            else "rgba(15,23,42,0.7)",
            colorbar=dict(
                title=dict(text=metric_name, side="right"),
                tickprefix="" if "PTI" in metric_name else "$",
                tickformat=",.2f" if "PTI" in metric_name else ",",
                ticksuffix="x" if "PTI" in metric_name else "",
                thickness=12,
                len=0.55,
                y=0.5,
                yanchor="middle",
                bgcolor="rgba(255,255,255,0.85)"
                if not is_dark_mode
                else "rgba(15,23,42,0.9)",
                borderwidth=0,
            ),
            hoverinfo="skip",
            showscale=True,
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=city_polygons_4326["center_lat"],
            lon=city_polygons_4326["center_lon"],
            mode="markers",
            marker=dict(size=30, opacity=0.0, color="rgba(0,0,0,0)"),
            customdata=city_polygons_4326[
                ["city", "metro_name", "avg_metric_value", "rank", "rank_total"]
            ].values,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        mapbox=dict(
            style=map_style,
            zoom=US_ZOOM_LEVEL,
            center={"lat": US_CENTER_LAT, "lon": US_CENTER_LON},
            bounds=US_BOUNDS,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=650,
        clickmode="event+select",
        dragmode="pan",
        hoverlabel=dict(
            bgcolor="white" if not is_dark_mode else "#020617",
            font_size=13,
            font_family="Arial",
        ),
    )

    return fig, city_polygons_4326

# ----------------- ZIP LEVEL -----------------
def create_zip_choropleth(
    gdf, map_style, city_coords, center_df, metric_name, is_dark_mode=False
):
    if gdf.empty:
        return None, None

    gdf = gdf[gdf["metric_value"].notna()].copy()
    if gdf.empty:
        st.warning(f"No valid data for {metric_name}")
        return None, None

    gdf = gdf.reset_index(drop=True)
    gdf["id"] = gdf.index.astype(str)
    gdf = compute_rankings(gdf, "metric_value", "zip_code_str")

    gdf_4326 = (
        gdf.to_crs(epsg=4326)
        if isinstance(gdf, gpd.GeoDataFrame) and gdf.crs and gdf.crs != "EPSG:4326"
        else gdf.copy()
    )

    if isinstance(gdf_4326, gpd.GeoDataFrame) and gdf_4326.geometry.notna().any():
        gdf_proj = gdf_4326.to_crs(epsg=2163)
        centroids_proj = gdf_proj.geometry.centroid
        centroids_4326 = gpd.GeoSeries(centroids_proj, crs=2163).to_crs(epsg=4326)
        gdf_4326["center_lat"] = centroids_4326.y
        gdf_4326["center_lon"] = centroids_4326.x
    else:
        gdf_4326["center_lat"] = center_df["lat"]
        gdf_4326["center_lon"] = center_df["lon"]

    geojson = json.loads(gdf_4326.to_json())

    if city_coords:
        center_lat, center_lon = city_coords
    elif center_df is not None and not center_df.empty:
        center_lat = center_df["lat"].mean()
        center_lon = center_df["lon"].mean()
    else:
        center_lat = gdf_4326["center_lat"].mean()
        center_lon = gdf_4326["center_lon"].mean()

    vmin = float(gdf["metric_value"].min())
    vmax = float(gdf["metric_value"].max())
    colorscale = get_colorscale(metric_name, is_dark_mode)

    fig = go.Figure()
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=gdf_4326["id"],
            z=gdf_4326["metric_value"],
            featureidkey="properties.id",
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
            marker_opacity=0.9,
            marker_line_width=0.5,
            marker_line_color="rgba(248,250,252,0.9)"
            if not is_dark_mode
            else "rgba(15,23,42,0.8)",
            selected=dict(marker=dict(opacity=1.0)),
            unselected=dict(marker=dict(opacity=0.35)),
            colorbar=dict(
                title=dict(text=metric_name, side="right"),
                tickprefix="" if "PTI" in metric_name else "$",
                tickformat=",.2f" if "PTI" in metric_name else ",",
                ticksuffix="x" if "PTI" in metric_name else "",
                thickness=12,
                len=0.55,
                y=0.5,
                yanchor="middle",
                bgcolor="rgba(255,255,255,0.85)"
                if not is_dark_mode
                else "rgba(15,23,42,0.9)",
                borderwidth=0,
            ),
            customdata=gdf_4326[
                ["zip_code_str", "city_full", "metric_value", "rank", "rank_total"]
            ].values,
            hovertemplate=(
                "<b>ZIP %{customdata[0]}</b><br>"
                "Metro: %{customdata[1]}<br>"
                + (
                    "PTI: %{customdata[2]:.2f}x"
                    if "PTI" in metric_name
                    else "Price: $%{customdata[2]:,.0f}"
                )
                + "<br>Rank: #%{customdata[3]} of %{customdata[4]}"
                + "<extra></extra>"
            ),
            showscale=True,
        )
    )

    fig.update_layout(
        mapbox=dict(
            style=map_style,
            zoom=9,
            center={"lat": center_lat, "lon": center_lon},
            bounds=US_BOUNDS,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=650,
        clickmode="event+select",
        dragmode="pan",
        hoverlabel=dict(
            bgcolor="white" if not is_dark_mode else "#020617",
            font_size=13,
            font_family="Arial",
        ),
    )

    return fig, gdf_4326

# ----------------- HISTORY CHART -----------------
def create_history_chart(zip_hist: pd.DataFrame, metro_avg: float, metric_name: str, is_dark_mode: bool = False):
    if zip_hist.empty:
        return None

    value_col = "PTI" if "PTI" in metric_name else "price"
    line_color = "#2563eb" if not is_dark_mode else "#60a5fa"
    avg_line_color = "#ea580c" if not is_dark_mode else "#fb923c"
    grid_color = "rgba(148,163,184,0.35)" if not is_dark_mode else "rgba(148,163,184,0.3)"
    text_color = "#111827" if not is_dark_mode else "#e5e7eb"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=zip_hist["year"],
            y=zip_hist[value_col],
            mode="lines+markers",
            name="This ZIP",
            line=dict(color=line_color, width=3),
            marker=dict(size=7, color=line_color),
            hovertemplate=(
                "Year: %{x}<br>"
                + (
                    "PTI: %{y:.2f}x"
                    if "PTI" in metric_name
                    else "Price: $%{y:,.0f}"
                )
                + "<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=metro_avg,
        line_dash="dash",
        line_color=avg_line_color,
        line_width=2,
        annotation_text=(
            f"Metro Avg: {metro_avg:.2f}x"
            if "PTI" in metric_name
            else f"Metro Avg: ${metro_avg:,.0f}"
        ),
        annotation_position="right",
        annotation_font_color=avg_line_color,
        annotation_font_size=10,
    )
    fig.update_layout(
        height=280,
        margin=dict(l=5, r=60, t=25, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="",
            gridcolor=grid_color,
            tickfont=dict(color=text_color, size=10),
            showline=True,
            linecolor=grid_color,
        ),
        yaxis=dict(
            title="",
            gridcolor=grid_color,
            tickfont=dict(color=text_color, size=10),
            tickprefix="" if "PTI" in metric_name else "$",
            tickformat=",.1f" if "PTI" in metric_name else ",.0f",
            showline=True,
            linecolor=grid_color,
        ),
        showlegend=False,
        hovermode="x unified",
    )
    return fig
