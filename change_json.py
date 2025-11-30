import geopandas as gpd

# 读取 ZIP shapefile
zcta = gpd.read_file("data/cb_2018_us_zcta510_500k.shp")

# 多边形简化（降低复杂度）
zcta_simplified = zcta.simplify(0.001, preserve_topology=True)

# 保存 GeoJSON
zcta_simplified.to_file("data/zcta_simplified.json", driver="GeoJSON")
cbsa = gpd.read_file("data/cb_2018_us_cbsa_500k.shp")
cbsa_simplified = cbsa.simplify(0.0005, preserve_topology=True)
cbsa_simplified.to_file("data/cbsa_simplified.json", driver="GeoJSON")
