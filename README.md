# 🏙️ Metro → ZIP Sale Price Explorer

An interactive Streamlit application for exploring U.S. housing affordability across metro areas and ZIP codes.

## 📊 What This App Visualizes

- **Median Sale Price** - Monthly median property sale prices
- **Price-to-Income Ratio (PTI)** - Affordability metrics
- **Historical Trends** - ZIP-level time series analysis
- **Metro → ZIP Drill-down** - Interactive navigation from metro to ZIP level
- **Multiple Map Styles** - Tile map, packed shapes, and CBSA polygons

This app is designed for fast exploration, data storytelling, and interactive real-estate analytics.

## 🚀 Live Demo

👉 Deploy on [Streamlit Community Cloud](https://rentsmapapp-36jh9xfygrdshmarfwlfod.streamlit.app) by connecting this GitHub repository.

## 📁 Project Structure

```
rents_map_app/
│
├── app.py                  # Main Streamlit app
├── charts.py               # Plotly chart builders
├── events.py               # Click/hover event utilities
├── geo_utils.py            # Shapefile loading + metro/ZIP matching
├── config_data.py          # Global configs + data loading logic
├── requirements.txt        # Python dependencies
│
├── data/
│   ├── house_ts_agg.csv    # Local aggregated sale price + income dataset
│   ├── cbsa_shapes.zip     # CBSA shapefile bundle (stored as ZIP)
│   └── zcta_shapes.zip     # ZCTA shapefile bundle (stored as ZIP)
│
└── .streamlit/
    ├── config.toml         # (optional) UI theme overrides
    └── secrets.toml        # (unused unless Databricks mode is enabled)
```

## 🗄️ Data Sources

### 1. Household Metrics (Local CSV)

**File:** `data/house_ts_agg.csv`

**Columns:**
- `city`, `city_full`, `zip_code`, `year`
- `median_sale_price`, `per_capita_income`
- `lat`, `lon`

### 2. CBSA Shapefile

**File:** `data/cbsa_shapes.zip`

**Contents (inside the ZIP):**
- `cb_2018_us_cbsa_500k.shp`
- `cb_2018_us_cbsa_500k.dbf`
- `cb_2018_us_cbsa_500k.shx`
- Additional supporting files

### 3. ZCTA Shapefile

**File:** `data/zcta_shapes.zip`

**Contents:**
- `cb_2018_us_zcta510_500k.shp`
- `cb_2018_us_zcta510_500k.dbf`
- `cb_2018_us_zcta510_500k.shx`
- Additional supporting files

## 📦 Shapefile Loading Logic

The app automatically loads ZIP shapefiles using:

```python
geopandas.read_file("zip://data/cbsa_shapes.zip")
```

This avoids GitHub's 100MB limit and makes the app fully self-contained for both local execution and Streamlit Cloud.

## ✨ Features

### 🗺️ Metro-Level Visualization

- Compare affordability across major U.S. metros
- **Multiple visualization modes:**
  - Hexbin metro tile map
  - Packed metro shapes
  - Real CBSA polygon map
- Hover tooltips with PTI / price metrics
- Ranking system (best → worst metros)
- Click any metro to drill down

### 📍 ZIP-Level Visualization

- Choropleth map of ZIP-level median sale price or PTI
- Rich hover tooltips
- Highlighted ZIP selection
- **Metro-level summary stats:**
  - Avg price/PTI
  - Best / worst ZIP
  - ZIP count
- Full history chart for selected ZIP

### 📈 Historical Trends

Each ZIP includes:
- Time series of median sale price
- Time series of PTI (Price-to-Income Ratio)
- Metro average overlay
- Tooltip with full numeric detail

## 📊 Metric Definitions

### Median Sale Price
Monthly median property sale price.

### Price-to-Income Ratio (PTI)
```
PTI = median_sale_price / per_capita_income
```
This is filtered to avoid unrealistic values (<0.5 or >50).

## 🧩 Data Loading Modes

### ✔ Local Data Mode (Default)

Set in `config_data.py`:
```python
USE_LOCAL_DATA = True
```

This will load:
- `data/house_ts_agg.csv`
- `data/cbsa_shapes.zip`
- `data/zcta_shapes.zip`

**Benefits:**
- No API keys required
- Works out-of-the-box on Streamlit Cloud

### ✔ Databricks Mode (Optional)

Set:
```python
USE_LOCAL_DATA = False
```

**Requires environment variables:**
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_WAREHOUSE_ID`

**This mode enables:**
- Live querying of Databricks Delta tables
- Direct integration with cloud-scale datasets

## 🌐 How Streamlit Cloud Loads Data

Streamlit Cloud automatically pulls:
- Your Python files
- Your `/data/*.csv`
- Your `/data/*.zip`
- Your dependencies from `requirements.txt`

**Important:**
Because shapefiles are in `data/*.zip`, no secrets are required and deployment is 100% portable.

## ▶️ Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🚢 Deploying to Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Select branch: `main`
4. Set main file: `app.py`

Streamlit will automatically:
- ✅ Install dependencies
- ✅ Load shapefile ZIPs
- ✅ Cache geospatial boundaries
- ✅ Run the app without needing secrets

## 🛠️ Tech Stack

- **Python** 3.9+
- **Streamlit** - Interactive web framework
- **Plotly / Mapbox** - Visualization and mapping
- **GeoPandas** - Geospatial data processing
- **Shapely** - Geometric operations
- **Pandas / NumPy** - Data manipulation

