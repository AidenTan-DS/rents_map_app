# Data Aggregation Documentation

This document explains how data is aggregated at different levels in the application, from raw monthly data to ZIP code level and metro level aggregations.

## Overview

The data aggregation pipeline follows this flow:

```
Raw Monthly Data (house_ts table)
  ↓
SQL Aggregation (month → year)
  ↓
Application-Level Aggregation (year → ZIP code)
  ↓
Metro-Level Aggregation (ZIP code → metro)
```

---

## 1. SQL-Level Aggregation (Month → Year)

### Location
- **Databricks Query**: `config_data.py` lines 212-229
- **User's SQL**: Creates `exported_house_ts_parquet` table

### SQL Query

```sql
SELECT
    h.city,
    h.city_full,
    h.zip_code,
    h.year,
    AVG(h.median_sale_price) AS median_sale_price,
    AVG(h.per_capita_income) AS per_capita_income,
    AVG(g.lat) AS lat,
    AVG(g.lon) AS lon
FROM workspace.data511.house_ts h
LEFT JOIN workspace.data511.zip_geo g
  ON CAST(h.zip_code AS INT) = CAST(g.zip_code AS INT)
WHERE h.median_sale_price IS NOT NULL
  AND h.median_sale_price > 0
GROUP BY
    h.city, h.city_full, h.zip_code, h.year
```

### Explanation

- **Purpose**: Aggregates monthly-level data to year-level data
- **Grouping**: `city`, `city_full`, `zip_code`, `year`
- **Aggregation Method**: `AVG()` (average) for all numeric columns
- **Result**: One record per `(city, zip_code, year)` combination

### Example

**Input** (monthly data in `house_ts` table):
```
ZIP 12345, Minneapolis, 2023-01 → median_sale_price: $500,000
ZIP 12345, Minneapolis, 2023-02 → median_sale_price: $510,000
ZIP 12345, Minneapolis, 2023-03 → median_sale_price: $520,000
...
ZIP 12345, Minneapolis, 2023-12 → median_sale_price: $550,000
```

**Output** (after SQL aggregation):
```
ZIP 12345, Minneapolis, 2023 → median_sale_price: $520,000 (average of 12 months)
```

---

## 2. ZIP Code Level Aggregation

### Location
- **File**: `app.py`
- **Lines**: 165-173 (PTI) or 190-198 (Median Sale Price)

### Code

#### For PTI (Price-to-Income Ratio):
```python
df_zip_metric = df_year.groupby(
    ["city", "city_full", "city_clean", "zip_code_str", "year"], 
    as_index=False
).agg(
    metric_value=("PTI", "mean"),
    lat=("lat", "mean"),
    lon=("lon", "mean"),
)
```

#### For Median Sale Price:
```python
df_zip_metric = df_year.groupby(
    ["city", "city_full", "city_clean", "zip_code_str", "year"], 
    as_index=False
).agg(
    metric_value=("median_sale_price", "mean"),
    lat=("lat", "mean"),
    lon=("lon", "mean"),
)
```

### Explanation

- **Purpose**: Ensures one record per ZIP code per year
- **Grouping**: `city`, `city_full`, `city_clean`, `zip_code_str`, `year`
- **Aggregation Method**: `mean()` (average)
- **Result**: One record per `(zip_code_str, year)` combination
- **Note**: If data is already at year level (from SQL), this aggregation is redundant but safe (no-op)

### Example

**Input** (`df_year` - already filtered by selected year):
```
ZIP 12345, Minneapolis, 2023 → PTI: 3.5
ZIP 12346, Minneapolis, 2023 → PTI: 4.0
ZIP 12347, Minneapolis, 2023 → PTI: 3.8
```

**Output** (`df_zip_metric`):
```
ZIP 12345, Minneapolis, 2023 → metric_value: 3.5
ZIP 12346, Minneapolis, 2023 → metric_value: 4.0
ZIP 12347, Minneapolis, 2023 → metric_value: 3.8
```

*(If data is already at year level, output is identical to input)*

---

## 3. Metro Level Aggregation

### Location
- **File**: `app.py`
- **Lines**: 175-182 (PTI) or 200-207 (Median Sale Price)

### Code

```python
df_city = df_zip_metric.groupby(
    ["city", "city_full", "city_clean"], 
    as_index=False
).agg(
    n=("zip_code_str", "count"),                    # Number of ZIP codes
    avg_metric_value=("metric_value", "mean"),      # Average across all ZIPs
    lat=("lat", "mean"),
    lon=("lon", "mean"),
)
```

### Explanation

- **Purpose**: Aggregates ZIP code level data to metro level
- **Grouping**: `city`, `city_full`, `city_clean` (metro only, not ZIP)
- **Aggregation Methods**:
  - `metric_value` → `mean()` (average of all ZIP codes in the metro)
  - `zip_code_str` → `count()` (number of ZIP codes in the metro)
- **Result**: One record per metro

### Example

**Input** (`df_zip_metric`):
```
ZIP 12345, Minneapolis, 2023 → metric_value: 3.5
ZIP 12346, Minneapolis, 2023 → metric_value: 4.0
ZIP 12347, Minneapolis, 2023 → metric_value: 3.8
ZIP 12348, Minneapolis, 2023 → metric_value: 4.2
ZIP 12349, Minneapolis, 2023 → metric_value: 3.9
```

**Output** (`df_city`):
```
Minneapolis, 2023 → 
  avg_metric_value: (3.5 + 4.0 + 3.8 + 4.2 + 3.9) / 5 = 3.88
  n: 5 (5 ZIP codes)
```

---

## 4. Metro Summary in ZIP View

### Location
- **File**: `app.py`
- **Lines**: 575-636

### Code

```python
# Get all ZIP codes for the selected metro
values = zip_df_city["metric_value"]

# Calculate Metro Average
metro_avg = values.mean()

# Calculate Max/Min (excluding zeros)
nonzero_values = values[values > 0]
max_value = nonzero_values.max()
min_value = nonzero_values.min()
```

### Explanation

- **Purpose**: Displays metro-level statistics in the ZIP view
- **Data Source**: `zip_df_city` (all ZIP codes in the selected metro for the selected year)
- **Calculations**:
  - **Metro Avg**: Average of all ZIP codes' `metric_value`
  - **Max**: Maximum `metric_value` among all ZIP codes
  - **Min**: Minimum `metric_value` among all ZIP codes (excluding zeros)
- **Consistency**: Uses the same aggregation logic as Metro View's `df_city`

### Example

**Input** (`zip_df_city` for Minneapolis, 2023):
```
ZIP 12345 → metric_value: 3.5
ZIP 12346 → metric_value: 4.0
ZIP 12347 → metric_value: 3.8
ZIP 12348 → metric_value: 4.2
ZIP 12349 → metric_value: 3.9
```

**Output** (Metro Summary):
```
Metro Avg: 3.88x
Max PTI: 4.2x
Min PTI: 3.5x
ZIP Codes (on map): 5
```

---

## 5. Time Series Chart Aggregation

### Location
- **File**: `app.py`
- **Lines**: 642-704

### Code

#### For PTI:
```python
# Step 1: Aggregate by ZIP and year (same as df_zip_metric calculation)
metro_zip_year = metro_hist_raw.groupby(
    ["city", "city_full", "city_clean", "zip_code_str", "year"], 
    as_index=False
).agg(PTI=("PTI", "mean"))

# Step 2: Filter to only include ZIPs that appear on the map
valid_zips_for_chart = zip_df_city["zip_code_str"].unique()
metro_zip_year = metro_zip_year[
    metro_zip_year["zip_code_str"].isin(valid_zips_for_chart)
]

# Step 3: Aggregate by year, averaging across all ZIPs
metro_hist = metro_zip_year.groupby("year", as_index=False).agg(
    PTI=("PTI", "mean")
).sort_values("year")
```

#### For Median Sale Price:
```python
# Step 1: Aggregate by ZIP and year
metro_zip_year = metro_hist_raw.groupby(
    ["city", "city_full", "city_clean", "zip_code_str", "year"], 
    as_index=False
).agg(metric_value=("median_sale_price", "mean"))

# Step 2: Filter to only include ZIPs that appear on the map
metro_zip_year = metro_zip_year[
    metro_zip_year["zip_code_str"].isin(valid_zips_for_chart)
]

# Step 3: Aggregate by year, averaging across all ZIPs
metro_hist = metro_zip_year.groupby("year", as_index=False).agg(
    metric_value=("metric_value", "mean")
).sort_values("year")
```

### Explanation

- **Purpose**: Creates time series data for metro-level trends
- **Two-Step Aggregation**:
  1. **Step 1**: Aggregate by ZIP and year (ensures one record per ZIP per year)
  2. **Step 2**: Aggregate by year, averaging across all ZIPs (metro-level average per year)
- **Filtering**: Only includes ZIP codes that appear on the map (`valid_zips_for_chart`)
- **Result**: One record per year, representing the average across all ZIP codes in the metro

### Example

**Input** (`metro_hist_raw` for Minneapolis, all years):
```
ZIP 12345, 2021 → PTI: 3.2
ZIP 12345, 2022 → PTI: 3.4
ZIP 12345, 2023 → PTI: 3.5
ZIP 12346, 2021 → PTI: 3.8
ZIP 12346, 2022 → PTI: 3.9
ZIP 12346, 2023 → PTI: 4.0
ZIP 12347, 2021 → PTI: 3.5
ZIP 12347, 2022 → PTI: 3.7
ZIP 12347, 2023 → PTI: 3.8
```

**After Step 1** (aggregate by ZIP and year - already done if data is at year level):
```
ZIP 12345, 2021 → PTI: 3.2
ZIP 12345, 2022 → PTI: 3.4
ZIP 12345, 2023 → PTI: 3.5
ZIP 12346, 2021 → PTI: 3.8
ZIP 12346, 2022 → PTI: 3.9
ZIP 12346, 2023 → PTI: 4.0
ZIP 12347, 2021 → PTI: 3.5
ZIP 12347, 2022 → PTI: 3.7
ZIP 12347, 2023 → PTI: 3.8
```

**After Step 2** (filter to ZIPs on map - assume all 3 ZIPs are on map):
```
(Same as Step 1 output)
```

**After Step 3** (aggregate by year, average across ZIPs):
```
2021 → PTI: (3.2 + 3.8 + 3.5) / 3 = 3.50
2022 → PTI: (3.4 + 3.9 + 3.7) / 3 = 3.67
2023 → PTI: (3.5 + 4.0 + 3.8) / 3 = 3.77
```

---

## Summary Table

| Level | Grouping Dimensions | Aggregation Method | Result |
|-------|-------------------|-------------------|--------|
| **SQL (Month → Year)** | `city`, `zip_code`, `year` | `AVG()` | 1 record per `(zip_code, year)` |
| **ZIP Code Level** | `city`, `zip_code_str`, `year` | `mean()` | 1 record per `(zip_code_str, year)` |
| **Metro Level** | `city` (metro only) | `mean()` across ZIPs | 1 record per metro |
| **Time Series** | `year` (after ZIP aggregation) | `mean()` across ZIPs | 1 record per year per metro |

---

## Key Points

1. **SQL Aggregation**: Monthly data is aggregated to year level using `AVG()` in SQL
2. **ZIP Code Aggregation**: Ensures one record per ZIP per year (redundant if SQL already aggregated)
3. **Metro Aggregation**: Averages all ZIP codes within a metro to get metro-level metrics
4. **Consistency**: Metro Summary and Time Series Chart use the same aggregation logic
5. **Filtering**: Time Series Chart only includes ZIP codes that appear on the map

---

## Notes

- If data is already at year level (from SQL aggregation), the ZIP code level aggregation in `app.py` is redundant but safe (no-op)
- All aggregations use `mean()` (average) method
- Metro-level metrics represent the average across all ZIP codes in the metro
- The time series chart ensures consistency by using the same ZIP filtering as the Metro Summary

