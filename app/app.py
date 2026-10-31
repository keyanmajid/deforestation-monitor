import json
from pathlib import Path

import folium
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

HERE = Path(__file__).parent
st.set_page_config(page_title="Deforestation Monitor", layout="wide")


@st.cache_data
def load():
    gdf = gpd.read_file(HERE / "changes.geojson")
    meta = json.loads((HERE / "meta.json").read_text())
    return gdf, meta


gdf, meta = load()

st.title("Deforestation Monitor: Machadinho d'Oeste, Rondônia")
st.caption("Sentinel-2 imagery · U-Net forest segmentation · 2022 to 2024 change detection")

with st.sidebar:
    st.header("Filters")
    min_drop = st.slider("Minimum NDVI drop (confidence)", 0.10, 0.60, 0.30, 0.05,
                         help="Higher = fewer, more reliable detections")
    min_area = st.slider("Minimum patch size (ha)", 0.5, 10.0, 0.5, 0.5)

sel = gdf[(gdf["ndvi_drop"] >= min_drop) & (gdf["area_ha"] >= min_area)]

c1, c2, c3 = st.columns(3)
c1.metric("Patches detected", len(sel))
c2.metric("Forest loss (ha)", f"{sel['area_ha'].sum():,.0f}")
c3.metric("Mean NDVI drop", f"{sel['ndvi_drop'].mean():.2f}" if len(sel) else "n/a")

m = folium.Map(location=meta["center"], zoom_start=12, tiles=None)
folium.TileLayer("OpenStreetMap", name="Street map", show=False).add_to(m)
folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Esri satellite", show=False).add_to(m)
for yr in ("2022", "2024"):
    folium.raster_layers.ImageOverlay(
        image=str(HERE / f"rgb_{yr}.png"), bounds=meta["bounds"],
        name=f"Sentinel-2 {yr}", show=(yr == "2024")).add_to(m)


def color(d):
    return "#d7191c" if d >= 0.4 else "#fdae61" if d >= 0.3 else "#ffff33"


if len(sel):
    folium.GeoJson(
        sel[["area_ha", "ndvi_drop", "geometry"]].__geo_interface__,
        name="Detected forest loss",
        style_function=lambda f: {
            "color": color(f["properties"]["ndvi_drop"]),
            "fillColor": color(f["properties"]["ndvi_drop"]),
            "weight": 2, "fillOpacity": 0.35},
        tooltip=folium.GeoJsonTooltip(fields=["area_ha", "ndvi_drop"],
                                      aliases=["Area (ha)", "NDVI drop"]),
    ).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, height=650, use_container_width=True, returned_objects=[])

st.download_button("Download filtered GeoJSON", sel.to_json(),
                   "deforestation_patches.geojson", "application/geo+json")
with st.expander("Patch table"):
    st.dataframe(sel.drop(columns="geometry").sort_values("area_ha", ascending=False),
                 use_container_width=True)

with st.expander("Method and honest limitations"):
    st.markdown("""
**Pipeline:** cloud-masked Sentinel-2 dry-season median composites (2022, 2024) →
U-Net (ResNet34 encoder) forest segmentation → probability-based change detection
(hysteresis thresholds, morphological opening, minimum area, NDVI-drop confirmation).

**Model:** trained on ESA WorldCover 2021 tree cover (noisy labels). Validation IoU 0.86 on a held-out
region, measuring agreement with WorldCover, not true forest.

**Check against Hansen Global Forest Change loss (2022-24):** with NDVI drop ≥ 0.3, roughly
60-70% of patches overlap Hansen loss, while only about 20% of Hansen's loss patches are recovered.
Small sample (59 patches), so treat figures as rough.

**Limitations:** most detections lie on land with earlier recorded disturbance (regrowth and pasture
edges), single study area, two dates only, dry-season composites only. This is a near-real-time
alert prototype, not a validated measurement of deforestation.
""")