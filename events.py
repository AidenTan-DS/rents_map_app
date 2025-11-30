# events.py
def extract_city_from_event(event):
    """Extract the city name from a metro-level selection event."""
    if event and event.selection and event.selection.points:
        clicked_point = event.selection.points[0]
        cd = clicked_point.get("customdata", None)
        if isinstance(cd, (list, tuple)) and len(cd) > 0:
            return cd[0]
    return None

def extract_zip_from_event(event, gdf_zip=None):
    """
    Extract the ZIP code string from a ZIP-level selection event.
    """
    if event and event.selection and event.selection.points:
        clicked_point = event.selection.points[0]
        cd = clicked_point.get("customdata", None)
        if isinstance(cd, (list, tuple)) and len(cd) > 0:
            return str(cd[0])
        location = clicked_point.get("location", None)
        if location is not None and gdf_zip is not None:
            match = gdf_zip[gdf_zip["id"] == str(location)]
            if not match.empty:
                return str(match.iloc[0]["zip_code_str"])
        point_idx = clicked_point.get("point_index", None)
        if point_idx is not None and gdf_zip is not None and point_idx < len(gdf_zip):
            return str(gdf_zip.iloc[point_idx]["zip_code_str"])
    return None
