import geopandas as gpd

# Convert CBSA to GeoJSON
cbsa = gpd.read_file("cb_2018_us_cbsa_500k.shp")
cbsa.to_file("cbsa.json", driver="GeoJSON")
print("Converted CBSA → cbsa.json")

# Convert ZCTA to GeoJSON
zcta = gpd.read_file("cb_2018_us_zcta510_500k.shp")
zcta.to_file("zcta.json", driver="GeoJSON")
print("Converted ZCTA → zcta.json")
