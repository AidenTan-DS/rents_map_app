# 🏙️ Metro → ZIP Sale Price Explorer

An interactive Streamlit application for exploring **US housing affordability** across metro areas and ZIP codes.  
The app visualizes **median sale prices** and **Price-to-Income Ratio (PTI)** using multiple map styles, including:

- 🟦 **Hexbin metro tile map**
- 🟩 **Packed CBSA metro shapes**
- 🗺️ **US basemap with real CBSA polygons**
- 📍 **ZIP-level choropleth maps**

This tool enables users to quickly evaluate housing affordability, compare metros, and drill down into neighborhood-level data.

---

## 🚀 Live Demo

👉 _You can deploy this app on Streamlit Community Cloud_

---

## 📊 Features

### **🗺️ Metro-Level Visualization**
- View affordability across 30+ major US metros  
- Multiple visualization styles:
  - **Hexbin map** (state-like tiles)
  - **Packed metro shapes** (abstract but space-efficient)
  - **Real CBSA polygons** on Mapbox
- Hover for metro-level stats
- Click any metro to drill down into ZIP-level details

---

### **📍 ZIP-Level Visualization**
- Visualize ZIP-level **median sale price** or **PTI**
- Interactive map with tooltips and click selection
- Historical trend line charts for each ZIP code
- Metro-wide summary statistics:
  - Average price/PTI
  - Max & Min ZIP
  - ZIP count

---

### **📊 Metrics Supported**
**1. Median Sale Price**  
Monthly median sale prices aggregated from Databricks tables.

**2. Price-to-Income Ratio (PTI)**  
A simple affordability measure:

