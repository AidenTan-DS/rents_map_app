# app.py
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd

from config_data import (
    get_dynamic_css,
    get_colorscale,
    load_all_data,
    compute_pti,
    compute_rankings,
    get_metro_yoy,
    US_BOUNDS,
    US_CENTER_LAT,
    US_CENTER_LON,
    US_ZOOM_LEVEL,
)
from geo_utils import load_cbsa_shapes, load_zcta_shapes, get_zip_polygons_for_metro
from charts import create_city_choropleth, create_zip_choropleth, create_history_chart, create_metro_timeseries_chart
from events import extract_city_from_event, extract_zip_from_event

# =========================================================================
# 1. Page config
# =========================================================================
st.set_page_config(
    page_title="Interactive Metro → ZIP Sale Price Map",
    layout="wide",
    initial_sidebar_state="collapsed",  # Collapse sidebar to maximize map space
)

# =========================================================================
# 2. Session state init
# =========================================================================
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "city"
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = None
if "selected_zip" not in st.session_state:
    st.session_state["selected_zip"] = None

# =========================================================================
# 3. Load data
# =========================================================================
try:
    df_all = load_all_data()
except FileNotFoundError as e:
    st.error(f"""
        ❌ **Data File Not Found**
        
        Unable to find the data file. Please check:
        - Data file exists: `data/house_ts_agg.csv`
        - File path is correct
        - You have read permissions
        
        **Technical details:** {str(e)}
    """)
    if st.button("🔄 Retry Loading Data"):
        st.cache_data.clear()
        st.rerun()
    st.stop()
except Exception as e:
    st.error(f"""
        ❌ **Data Loading Error**
        
        Unable to load data from the source. Please check:
        - Data file format is correct
        - File is not corrupted
        - You have read permissions
        
        **Technical details:** {str(e)}
    """)
    if st.button("🔄 Retry Loading Data"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

if df_all.empty:
    st.warning("""
        ⚠️ **No Data Available**
        
        The data file is empty or contains no valid records.
        Please check the data source and try again.
    """)
    st.stop()

min_year = int(df_all["year"].min())
max_year = int(df_all["year"].max())

# Detect Streamlit theme (for UI elements, not just map style)
# This ensures text and UI elements are visible regardless of map style
try:
    # Get theme from Streamlit config
    theme_base = st.get_option("theme.base")
    streamlit_is_dark = theme_base == "dark"
except:
    # If detection fails, default to light theme
    streamlit_is_dark = False

# Use Streamlit theme for UI elements (text, backgrounds, charts)
# Map style only affects the map tiles themselves, not UI colors
is_dark_mode = streamlit_is_dark

# Map style is fixed to light
map_style = "carto-positron"

# =========================================================================
# 4. Top Control Panel (expanded layout)
# =========================================================================
st.markdown("### 🧭 Control Panel")
control_col1, control_col2, control_col3 = st.columns([2, 2, 3])

with control_col1:
    # Year slider
    selected_year = st.slider("Year", min_year, max_year, max_year, help=f"Data range: {min_year} – {max_year}")

with control_col2:
    # Metric (compact horizontal layout)
    metric_type = st.radio(
        "Metric",
        ["Price-to-Income Ratio (PTI)", "Median Sale Price"],
        index=0,
        horizontal=True,
        help="PTI: affordability (lower = more affordable)\nPrice: median home sale price",
    )

with control_col3:
    # Metro search in city view
    if st.session_state["view_mode"] == "city":
        st.markdown("**🔍 Quick Metro Search**")
        df_filtered_sidebar = df_all[df_all["year"] == selected_year].copy()
        if not df_filtered_sidebar.empty:
            df_city_sidebar = (
                df_filtered_sidebar.groupby(["city", "city_full"], as_index=False)
                .agg(avg_median_sale_price=("median_sale_price", "mean"))
            )
            metro_list = (
                df_city_sidebar.drop_duplicates(subset=["city_full"])
                .sort_values("city_full")["city_full"]
                .tolist()
            )

            # Use index=None so no option is selected initially, placeholder shows "Type to search..."
            selected_metro = st.selectbox(
                "Select metro",
                metro_list,
                index=None,
                placeholder="Type to search...",
                format_func=lambda x: f"📍 {x}",
                label_visibility="collapsed",
            )

            # Auto-navigate to ZIP view when a metro is selected
            if selected_metro:
                city_match = (
                    df_city_sidebar[df_city_sidebar["city_full"] == selected_metro]["city"].iloc[0]
                )
                # Only navigate if the selection changed
                if st.session_state.get("selected_city") != city_match:
                    st.session_state["selected_city"] = city_match
                    st.session_state["view_mode"] = "zip"
                    st.session_state["selected_zip"] = None
                    st.rerun()

st.markdown("---")

# =========================================================================
# 5. Apply CSS
# =========================================================================
st.markdown(get_dynamic_css(is_dark_mode), unsafe_allow_html=True)

# =========================================================================
# 6. Build metric data for selected_year
# =========================================================================
df_year = df_all[df_all["year"] == selected_year].copy()
if df_year.empty:
    st.warning(f"### ⚠️ No data available for {selected_year}")
    st.stop()

if metric_type == "Price-to-Income Ratio (PTI)":
    df_year = compute_pti(df_year)
    value_source_col = "PTI"
    if df_year.empty:
        st.warning(f"⚠️ PTI values out of range for {selected_year}.")
        st.stop()

    df_zip_metric = (
        df_year.groupby(
            ["city", "city_full", "city_clean", "zip_code_str", "year"], as_index=False
        ).agg(
            metric_value=("PTI", "mean"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
    )

    df_city = (
        df_zip_metric.groupby(["city", "city_full", "city_clean"], as_index=False).agg(
            n=("zip_code_str", "count"),
            avg_metric_value=("metric_value", "mean"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
    )
else:
    df_year = df_year[df_year["median_sale_price"].notna()].copy()
    value_source_col = "median_sale_price"
    if df_year.empty:
        st.warning(f"⚠️ No valid price data for {selected_year}.")
        st.stop()

    df_zip_metric = (
        df_year.groupby(
            ["city", "city_full", "city_clean", "zip_code_str", "year"], as_index=False
        ).agg(
            metric_value=("median_sale_price", "mean"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
    )

    df_city = (
        df_zip_metric.groupby(["city", "city_full", "city_clean"], as_index=False).agg(
            n=("zip_code_str", "count"),
            avg_metric_value=("metric_value", "mean"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
    )

df_city_map = df_city.copy().reset_index(drop=True)
df_city_map = compute_rankings(df_city_map, "avg_metric_value", "city")

metro_yoy = get_metro_yoy(df_all, selected_year, metric_type)

# =========================================================================
# 7. Header & intro
# =========================================================================
st.title("🏙️ Metro → ZIP Sale Price/PTI Explorer")
st.caption(f"Year: **{selected_year}** · Metric: **{metric_type}**")

with st.expander("ℹ️ How to use this app", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ### **Navigation**
            - Hover over metros / ZIPs to preview basic statistics  
            - Use the **Year selector** in the control panel to choose which year to visualize  
            - Click a **metro** to zoom in and view its ZIP code map  
            - Click a **ZIP code** to view detailed metrics for the selected year  
            - Use **Back to All Metros** button to return to the national view  
            """
        )

    with col2:
        st.markdown(
            """
        **Metrics**
        - **Median Sale Price**  
          - Metro view: average of ZIP-level *monthly median* sale prices (selected year)  
          - ZIP view: average *monthly median* sale price for this ZIP (selected year)  
        - **PTI (Price-to-Income Ratio)** = Price ÷ (Income × 2.51)  
          - Where 2.51 is the median household size  
          - Metro view: average PTI across ZIPs in the metro  
          - ZIP view: PTI for this ZIP  
          - Lower PTI = more affordable
        """
        )

current_metro_name = None
if st.session_state["selected_city"]:
    row_sel = df_city_map[df_city_map["city"] == st.session_state["selected_city"]]
    if not row_sel.empty:
        current_metro_name = row_sel["city_full"].iloc[0]

st.markdown("---")

# =========================================================================
# 8. Main view (metro / zip)
# =========================================================================
if st.session_state["view_mode"] == "city":
    # --------------------- METRO VIEW ---------------------
    st.info(
        f"📍 **Metro View ({selected_year}) · {metric_type}**  · "
        f"Hover for details · Click to drill down · Scroll to zoom"
    )

    st.markdown("#### 📊 National Summary")
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)

    with col_s1:
        st.metric("Total Metros", len(df_city_map))

    with col_s2:
        avg_val = df_city_map["avg_metric_value"].mean()
        if metric_type == "Price-to-Income Ratio (PTI)":
            st.metric("Avg PTI", f"{avg_val:.2f}x")
        else:
            st.metric("Avg Price", f"${avg_val:,.0f}")

    with col_s3:
        top_metro = df_city_map.loc[df_city_map["avg_metric_value"].idxmax()]
        metro_label_high = top_metro["city_full"]
        if metric_type == "Price-to-Income Ratio (PTI)":
            st.metric("Highest PTI", f"{top_metro['avg_metric_value']:.2f}x")
        else:
            st.metric("Highest Price", f"${top_metro['avg_metric_value']:,.0f}")
        st.caption(f"Metro: **{metro_label_high}**")

    with col_s4:
        bottom_metro = df_city_map.loc[df_city_map["avg_metric_value"].idxmin()]
        metro_label_low = bottom_metro["city_full"]
        if metric_type == "Price-to-Income Ratio (PTI)":
            st.metric("Lowest PTI", f"{bottom_metro['avg_metric_value']:.2f}x")
        else:
            st.metric("Lowest Price", f"${bottom_metro['avg_metric_value']:,.0f}")
        st.caption(f"Metro: **{metro_label_low}**")

    with col_s5:
        if not metro_yoy.empty and "yoy_pct" in metro_yoy.columns:
            avg_yoy = metro_yoy["yoy_pct"].mean()
            if not pd.isna(avg_yoy):
                st.metric(
                    "Avg YoY Change",
                    f"{avg_yoy:+.1f}%",
                    delta="vs last year",
                    delta_color="off",
                )
            else:
                st.metric("Avg YoY Change", "N/A", delta="No prior year", delta_color="off")
        else:
            st.metric("Avg YoY Change", "N/A", delta="No prior year", delta_color="off")

    st.markdown("---")

    fig_city = None
    gdf_metro = None
    try:
        with st.spinner("🗺️ Loading metro boundaries..."):
            cbsa_shapes = load_cbsa_shapes()
        with st.spinner("📊 Generating metro map..."):
            fig_city, gdf_metro = create_city_choropleth(
                df_city_map, cbsa_shapes, map_style, metric_type, is_dark_mode
            )
    except Exception as e:
        st.error(f"""
            ❌ **Map Generation Error**
            
            Unable to generate the metro map. This could be due to:
            - Missing shapefile data
            - Corrupted shapefile files
            - Insufficient memory
            
            **Technical details:** {str(e)}
        """)

    if fig_city is not None and gdf_metro is not None:
        event = st.plotly_chart(
            fig_city,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key=f"metro_map_{selected_year}_{metric_type}_{map_style}",
            config={"scrollZoom": True},
        )
        clicked_city = extract_city_from_event(event)
        if clicked_city and clicked_city != st.session_state["selected_city"]:
            st.session_state["selected_city"] = clicked_city
            st.session_state["selected_zip"] = None
            st.session_state["view_mode"] = "zip"
            st.rerun()

else:
    # --------------------- ZIP VIEW ---------------------
    selected_city = st.session_state["selected_city"]
    if not selected_city:
        st.warning("⚠️ No metro selected. Please use the sidebar search to select a metro.")
        st.stop()

    st.markdown(f"### 🗺️ `USA` → `{current_metro_name or selected_city}` → `ZIP Codes`")
    
    # Back to All Metros button between navigation and info box
    if st.button("⬅️ Back to All Metros", use_container_width=False):
        st.session_state["view_mode"] = "city"
        st.session_state["selected_city"] = None
        st.session_state["selected_zip"] = None
        st.rerun()
    
    st.markdown("---")

    label = current_metro_name or selected_city
    st.info(
        f"📍 **{label}** ({selected_year}) · {metric_type}  · "
        f"Click ZIPs to see details · Scroll to zoom"
    )

    try:
        with st.spinner("🗺️ Loading ZIP code boundaries..."):
            zcta_shapes = load_zcta_shapes()
        with st.spinner("📊 Processing ZIP code data..."):
            zip_df_city, gdf_merge = get_zip_polygons_for_metro(
                selected_city, zcta_shapes, df_zip_metric
            )
    except Exception as e:
        st.error(f"""
            ❌ **ZIP Code Data Error**
            
            Unable to load ZIP code boundaries. This could be due to:
            - Missing shapefile data
            - Corrupted shapefile files
            - Data processing error
            
            **Technical details:** {str(e)}
        """)
        zip_df_city, gdf_merge = pd.DataFrame(), gpd.GeoDataFrame()

    if gdf_merge.empty or zip_df_city.empty:
        st.warning(f"""
            ⚠️ **No ZIP Code Data Available**
            
            No ZIP code data found for **{selected_city}** in **{selected_year}**.
            
            **Suggestions:**
            - Try selecting a different year
            - Try selecting a different metro area
            - Check if data exists for this metro in other years
        """)
    else:
        # Only keep ZIPs that have metric values
        zip_df_city = zip_df_city[zip_df_city["metric_value"].notna()].copy()

        # Only keep ZIPs that appear in the polygon GeoDataFrame (consistency with the map)
        valid_zips = gdf_merge["zip_code_str"].unique()
        zip_df_city = zip_df_city[zip_df_city["zip_code_str"].isin(valid_zips)].copy()

        if zip_df_city.empty:
            st.warning(f"""
                ⚠️ **No Valid Data Available**
                
                No valid {metric_type} data for **{selected_city}** in **{selected_year}**.
                
                **Possible reasons:**
                - Data not available for this metro/year combination
                - All values are missing or invalid
                - Data filtering removed all records
            """)
        else:
            # Rankings at ZIP level (within this metro)
            zip_df_city = compute_rankings(zip_df_city, "metric_value", "zip_code_str")

            if st.session_state.get("selected_zip") is None and not zip_df_city.empty:
                st.session_state["selected_zip"] = zip_df_city["zip_code_str"].iloc[0]

            col_map, col_detail = st.columns([2, 1.2])  # Wider detail panel for better plot visibility

            with col_map:
                city_coords = None
                with st.spinner("📊 Generating ZIP code map..."):
                    fig_zip, gdf_zip = create_zip_choropleth(
                        gdf_merge, map_style, city_coords, zip_df_city, metric_type, is_dark_mode
                    )
                if fig_zip is not None and gdf_zip is not None:
                    event = st.plotly_chart(
                        fig_zip,
                        width="stretch",
                        on_select="rerun",
                        selection_mode="points",
                        key=f"zip_map_{selected_city}_{selected_year}_{metric_type}_{map_style}",
                        config={"scrollZoom": True},
                    )
                    clicked_zip = extract_zip_from_event(event, gdf_zip)
                    if clicked_zip:
                        st.session_state["selected_zip"] = clicked_zip

            with col_detail:
                st.subheader("📋 ZIP Details")
                active_zip = st.session_state.get("selected_zip")
                if not active_zip:
                    st.info("👈 Click any ZIP on the map")
                else:
                    row_now = zip_df_city[zip_df_city["zip_code_str"] == active_zip]
                    if row_now.empty:
                        st.warning(f"⚠️ No data for ZIP {active_zip}")
                    else:
                        metric_val = float(row_now["metric_value"].iloc[0])
                        metro_avg_now = float(zip_df_city["metric_value"].mean())
                        diff = metric_val - metro_avg_now
                        pct_diff = (diff / metro_avg_now * 100) if metro_avg_now != 0 else 0.0
                        rank = int(row_now["rank"].iloc[0])
                        rank_total = int(row_now["rank_total"].iloc[0])
                        percentile = float(row_now["percentile"].iloc[0])
                        metro_name = row_now["city_full"].iloc[0]

                        # Set title colors based on theme
                        title_color = "#e5e7eb" if is_dark_mode else "#111827"
                        caption_color = "#9ca3af" if is_dark_mode else "#6b7280"
                        
                        st.markdown(
                            f'<h3 style="color: {title_color};">ZIP <code style="background: {"rgba(255,255,255,0.1)" if is_dark_mode else "rgba(0,0,0,0.05)"}; padding: 2px 6px; border-radius: 4px;">{active_zip}</code></h3>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<p style="color: {caption_color}; font-size: 0.875rem; margin-top: -0.5rem;">{metro_name}</p>',
                            unsafe_allow_html=True
                        )

                        # YoY for this ZIP
                        if metric_type == "Price-to-Income Ratio (PTI)":
                            zip_prev_raw = df_all[
                                (df_all["city"] == selected_city)
                                & (df_all["zip_code_str"] == active_zip)
                                & (df_all["year"] == selected_year - 1)
                            ].copy()
                            zip_prev_raw = compute_pti(zip_prev_raw) if not zip_prev_raw.empty else pd.DataFrame()
                            if not zip_prev_raw.empty:
                                prev_val = zip_prev_raw["PTI"].mean()
                                yoy_change = ((metric_val - prev_val) / prev_val * 100)
                                main_value = f"{metric_val:.2f}x"
                                delta_text = f"{yoy_change:+.1f}% YoY"
                            else:
                                main_value = f"{metric_val:.2f}x"
                                delta_text = "No prior year"
                        else:
                            zip_prev = df_all[
                                (df_all["city"] == selected_city)
                                & (df_all["zip_code_str"] == active_zip)
                                & (df_all["year"] == selected_year - 1)
                                & df_all["median_sale_price"].notna()
                            ]
                            if not zip_prev.empty:
                                prev_val = zip_prev["median_sale_price"].mean()
                                yoy_change = ((metric_val - prev_val) / prev_val * 100)
                                main_value = f"${metric_val:,.0f}"
                                delta_text = f"{yoy_change:+.1f}% YoY"
                            else:
                                main_value = f"${metric_val:,.0f}"
                                delta_text = "No prior year"

                        rank_percentile = 100 - percentile
                        if pct_diff > 5:
                            diff_label = f"{pct_diff:+.1f}% above metro avg"
                        elif pct_diff < -5:
                            diff_label = f"{pct_diff:+.1f}% below metro avg"
                        else:
                            diff_label = f"{pct_diff:+.1f}% vs metro avg"

                        # Set text colors based on theme
                        label_color = "#9ca3af" if is_dark_mode else "#6b7280"  # Lighter gray for dark mode
                        text_color = "#e5e7eb" if is_dark_mode else "#111827"  # Light text for dark mode, dark for light
                        secondary_text_color = "#d1d5db" if is_dark_mode else "#4b5563"  # Secondary text color
                        
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div style="font-size: 0.8rem; text-transform: uppercase; color: {label_color}; margin-bottom: 0.25rem;">
                                    {'PTI Ratio' if 'PTI' in metric_type else 'Median Sale Price'}
                                </div>
                                <div style="font-size: 1.6rem; font-weight: 600; margin-bottom: 0.1rem; color: {text_color};">
                                    {main_value}
                                </div>
                                <div style="font-size: 0.85rem; color: {secondary_text_color}; margin-bottom: 0.6rem;">
                                    {delta_text}
                                </div>
                                <div style="font-size: 0.9rem; color: {text_color}; line-height: 1.5;">
                                    <b>Rank:</b> #{rank} of {rank_total} (descending) · Top {rank_percentile:.0f}% in this metro<br>
                                    <b>Relative to metro:</b> {diff_label}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("#### 📈 Trend")
                        if metric_type == "Price-to-Income Ratio (PTI)":
                            zip_hist_raw = df_all[
                                (df_all["city"] == selected_city)
                                & (df_all["zip_code_str"] == active_zip)
                            ].copy()
                            zip_hist_raw = compute_pti(zip_hist_raw)
                            if not zip_hist_raw.empty:
                                zip_hist = (
                                    zip_hist_raw.groupby("year", as_index=False)
                                    .agg(PTI=("PTI", "mean"))
                                    .sort_values("year")
                                )
                                if not zip_hist.empty:
                                    fig_hist = create_history_chart(
                                        zip_hist, metro_avg_now, metric_type, is_dark_mode
                                    )
                                    if fig_hist:
                                        st.plotly_chart(
                                            fig_hist,
                                            width="stretch",
                                            config={"displayModeBar": False},
                                        )
                                else:
                                    st.caption("No historical data for this ZIP.")
                            else:
                                st.caption("No historical data for this ZIP.")
                        else:
                            zip_hist = (
                                df_all[
                                    (df_all["city"] == selected_city)
                                    & (df_all["zip_code_str"] == active_zip)
                                    & df_all["median_sale_price"].notna()
                                ]
                                .groupby("year", as_index=False)
                                .agg(price=("median_sale_price", "mean"))
                                .sort_values("year")
                            )
                            if not zip_hist.empty:
                                fig_hist = create_history_chart(
                                    zip_hist, metro_avg_now, metric_type, is_dark_mode
                                )
                                if fig_hist:
                                    st.plotly_chart(
                                        fig_hist,
                                        width="stretch",
                                        config={"displayModeBar": False},
                                    )
                            else:
                                st.caption("No historical data for this ZIP.")

                        st.markdown("---")
                        csv = zip_df_city[
                            ["zip_code_str", "year", "metric_value", "city_full", "rank"]
                        ].to_csv(index=False)
                        st.download_button(
                            label="📥 Download ZIP-level data (CSV)",
                            data=csv,
                            file_name=f"{selected_city.replace(',', '_')}_{selected_year}_zipdata.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

            st.markdown("---")
            st.markdown("#### 📊 Metro Summary")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

            values = zip_df_city["metric_value"]
            nonzero_values = values[values > 0]

            with col_m1:
                st.metric("ZIP Codes (on map)", len(zip_df_city))

            with col_m2:
                if metric_type == "Price-to-Income Ratio (PTI)":
                    st.metric("Metro Avg", f"{values.mean():.2f}x")
                else:
                    st.metric("Metro Avg", f"${values.mean():,.0f}")

            with col_m3:
                if metric_type == "Price-to-Income Ratio (PTI)":
                    st.metric(
                        "Max PTI",
                        f"{nonzero_values.max():.2f}x"
                        if not nonzero_values.empty
                        else "N/A",
                    )
                else:
                    st.metric(
                        "Max Price",
                        f"${nonzero_values.max():,.0f}"
                        if not nonzero_values.empty
                        else "N/A",
                    )

            with col_m4:
                if metric_type == "Price-to-Income Ratio (PTI)":
                    st.metric(
                        "Min PTI",
                        f"{nonzero_values.min():.2f}x"
                        if not nonzero_values.empty
                        else "N/A",
                    )
                else:
                    st.metric(
                        "Min Price",
                        f"${nonzero_values.min():,.0f}"
                        if not nonzero_values.empty
                        else "N/A",
                    )

            with col_m5:
                metro_row = (
                    metro_yoy[metro_yoy["city"] == selected_city]
                    if not metro_yoy.empty
                    else pd.DataFrame()
                )
                if not metro_row.empty and "yoy_pct" in metro_row.columns:
                    yoy_val = metro_row["yoy_pct"].iloc[0]
                    if not pd.isna(yoy_val):
                        st.metric("YoY Change", f"{yoy_val:+.1f}%")
                    else:
                        st.metric("YoY Change", "N/A")
                else:
                    st.metric("YoY Change", "N/A")
            
            # Metro time series chart
            st.markdown("#### 📈 Change Over Time")
            # Use the same ZIP filtering as Metro Summary (only ZIPs that appear on the map)
            valid_zips_for_chart = zip_df_city["zip_code_str"].unique()
            
            if metric_type == "Price-to-Income Ratio (PTI)":
                # Get metro-level PTI data over time - use same calculation as Metro Summary
                # First aggregate by ZIP and year, then average across ZIPs for each year
                # AND filter to only include ZIPs that appear on the map (same as Metro Summary)
                metro_hist_raw = df_all[df_all["city"] == selected_city].copy()
                metro_hist_raw = compute_pti(metro_hist_raw)
                if not metro_hist_raw.empty:
                    # First aggregate by ZIP and year (same as df_zip_metric calculation)
                    metro_zip_year = (
                        metro_hist_raw.groupby(
                            ["city", "city_full", "city_clean", "zip_code_str", "year"], as_index=False
                        ).agg(PTI=("PTI", "mean"))
                    )
                    # Filter to only include ZIPs that appear on the map (same as Metro Summary)
                    metro_zip_year = metro_zip_year[metro_zip_year["zip_code_str"].isin(valid_zips_for_chart)].copy()
                    # Then average across ZIPs for each year (same as Metro Summary)
                    metro_hist = (
                        metro_zip_year.groupby("year", as_index=False)
                        .agg(PTI=("PTI", "mean"))
                        .sort_values("year")
                    )
                    if not metro_hist.empty:
                        fig_metro_ts = create_metro_timeseries_chart(
                            metro_hist, metric_type, is_dark_mode
                        )
                        if fig_metro_ts:
                            # Ensure the chart is rendered with correct theme
                            st.plotly_chart(
                                fig_metro_ts,
                                width="stretch",
                                config={
                                    "displayModeBar": False,
                                    "staticPlot": False,
                                },
                                use_container_width=True,
                            )
                    else:
                        st.caption("No historical data available for this metro.")
                else:
                    st.caption("No historical data available for this metro.")
            else:
                # Get metro-level price data over time - use same calculation as Metro Summary
                # First aggregate by ZIP and year, then average across ZIPs for each year
                # AND filter to only include ZIPs that appear on the map (same as Metro Summary)
                metro_hist_raw = df_all[
                    (df_all["city"] == selected_city)
                    & df_all["median_sale_price"].notna()
                ].copy()
                if not metro_hist_raw.empty:
                    # First aggregate by ZIP and year (same as df_zip_metric calculation)
                    metro_zip_year = (
                        metro_hist_raw.groupby(
                            ["city", "city_full", "city_clean", "zip_code_str", "year"], as_index=False
                        ).agg(metric_value=("median_sale_price", "mean"))
                    )
                    # Filter to only include ZIPs that appear on the map (same as Metro Summary)
                    metro_zip_year = metro_zip_year[metro_zip_year["zip_code_str"].isin(valid_zips_for_chart)].copy()
                    # Then average across ZIPs for each year (same as Metro Summary)
                    metro_hist = (
                        metro_zip_year.groupby("year", as_index=False)
                        .agg(metric_value=("metric_value", "mean"))
                        .sort_values("year")
                    )
                    if not metro_hist.empty:
                        fig_metro_ts = create_metro_timeseries_chart(
                            metro_hist, metric_type, is_dark_mode
                        )
                        if fig_metro_ts:
                            # Ensure the chart is rendered with correct theme
                            st.plotly_chart(
                                fig_metro_ts,
                                width="stretch",
                                config={
                                    "displayModeBar": False,
                                    "staticPlot": False,
                                },
                                use_container_width=True,
                            )
                    else:
                        st.caption("No historical data available for this metro.")
                else:
                    st.caption("No historical data available for this metro.")
