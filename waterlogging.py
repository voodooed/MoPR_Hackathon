# ==========================================================
# WATERLOGGING + DRAINAGE PIPELINE (ROBUST VERSION)
# ==========================================================

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from whitebox.whitebox_tools import WhiteboxTools

# ==========================================================
# CONFIG
# ==========================================================
INPUT_DEM = "DTM.tif"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Absolute paths
INPUT_DEM = os.path.abspath(INPUT_DEM)
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)

print("Input DEM:", INPUT_DEM)
print("Output dir:", OUTPUT_DIR)

# ==========================================================
# INIT WHITEBOX
# ==========================================================
wbt = WhiteboxTools()
wbt.set_working_dir(OUTPUT_DIR)
wbt.verbose = True

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def check_file(path, msg):
    if not os.path.exists(path):
        raise RuntimeError(f"❌ {msg}: {path}")

def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

def normalize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

# ==========================================================
# STEP 1: FILL DEPRESSIONS
# ==========================================================
filled = os.path.join(OUTPUT_DIR, "dtm_filled.tif")

wbt.fill_depressions(
    dem=INPUT_DEM,
    output=filled
)

check_file(filled, "FillDepressions failed")
print("✔ Filled DEM")

# ==========================================================
# STEP 2: FLOW DIRECTION
# ==========================================================
flow_dir = os.path.join(OUTPUT_DIR, "flow_dir.tif")

wbt.d8_pointer(
    dem=filled,
    output=flow_dir
)

check_file(flow_dir, "Flow direction failed")
print("✔ Flow direction")

# ==========================================================
# STEP 3: FLOW ACCUMULATION
# ==========================================================
flow_acc = os.path.join(OUTPUT_DIR, "flow_acc.tif")

wbt.d8_flow_accumulation(
    i=filled,
    output=flow_acc,
    out_type="cells"
)

check_file(flow_acc, "Flow accumulation failed")
print("✔ Flow accumulation")

# ==========================================================
# STEP 4: SLOPE
# ==========================================================
slope = os.path.join(OUTPUT_DIR, "slope.tif")

wbt.slope(
    dem=filled,
    output=slope,
    units="degrees"
)

check_file(slope, "Slope failed")
print("✔ Slope")

# ==========================================================
# STEP 5: TWI
# ==========================================================
twi = os.path.join(OUTPUT_DIR, "twi.tif")

wbt.wetness_index(
    sca=flow_acc,
    slope=slope,
    output=twi
)

check_file(twi, "TWI failed")
print("✔ TWI")

# ==========================================================
# STEP 6: STREAMS + HAND
# ==========================================================
streams = os.path.join(OUTPUT_DIR, "streams.tif")
hand = os.path.join(OUTPUT_DIR, "hand.tif")

wbt.extract_streams(
    flow_accum=flow_acc,
    output=streams,
    threshold=1000
)

check_file(streams, "Stream extraction failed")

wbt.elevation_above_stream(
    dem=filled,
    streams=streams,
    output=hand
)

check_file(hand, "HAND failed")
print("✔ Streams + HAND")

# ==========================================================
# STEP 7: WATERLOGGING INDEX
# ==========================================================
dem = load_raster(filled)
slope_arr = load_raster(slope)
twi_arr = load_raster(twi)
hand_arr = load_raster(hand)

twi_n = normalize(twi_arr)
slope_n = 1 - normalize(slope_arr)
elev_n = 1 - normalize(dem)
hand_n = 1 - normalize(hand_arr)

index = (twi_n + slope_n + elev_n + hand_n) / 4
index[np.isnan(dem)] = np.nan

threshold = np.nanpercentile(index, 70)
hotspots = index >= threshold

index_tif = os.path.join(OUTPUT_DIR, "waterlogging_index.tif")
hotspot_tif = os.path.join(OUTPUT_DIR, "hotspots.tif")

with rasterio.open(filled) as src:
    meta = src.meta.copy()

meta.update(dtype="float32", nodata=np.nan)

with rasterio.open(index_tif, "w", **meta) as dst:
    dst.write(index.astype("float32"), 1)

meta.update(dtype="uint8", nodata=0)

with rasterio.open(hotspot_tif, "w", **meta) as dst:
    dst.write(hotspots.astype("uint8"), 1)

print("✔ Waterlogging index + hotspots")

# ==========================================================
# STEP 8: COST SURFACE
# ==========================================================
streams_arr = load_raster(streams)

streams_bin = streams_arr > 0
hotspots_bin = hotspots == 1

unconnected = hotspots_bin & (~streams_bin)

cost = (
    0.5 * slope_n +
    0.3 * hand_n +
    0.2 * elev_n
)

cost[np.isnan(cost)] = 9999

cost_tif = os.path.join(OUTPUT_DIR, "cost.tif")
targets_tif = os.path.join(OUTPUT_DIR, "targets.tif")

meta.update(dtype="float32", nodata=9999)

with rasterio.open(cost_tif, "w", **meta) as dst:
    dst.write(cost.astype("float32"), 1)

meta.update(dtype="int32", nodata=0)

with rasterio.open(targets_tif, "w", **meta) as dst:
    dst.write(unconnected.astype("int32"), 1)

print("✔ Cost + targets")

# ==========================================================
# STEP 9: COST DISTANCE
# ==========================================================
backlink = os.path.join(OUTPUT_DIR, "backlink.tif")
cost_dist = os.path.join(OUTPUT_DIR, "cost_dist.tif")

wbt.cost_distance(
    source=streams,
    cost=cost_tif,
    out_accum=cost_dist,
    out_backlink=backlink
)

check_file(backlink, "CostDistance failed")
print("✔ Cost distance")

# ==========================================================
# STEP 10: COST PATHWAY
# ==========================================================
drain = os.path.join(OUTPUT_DIR, "drain_alternative.tif")

wbt.cost_pathway(
    destination=targets_tif,
    backlink=backlink,
    output=drain
)

check_file(drain, "CostPathway failed")
print("✔ Drainage paths")

# ==========================================================
# STEP 11: VISUALIZATION
# ==========================================================
drains = load_raster(drain)
# FIX: cost_pathway fills every "off-path" cell with the SAME sentinel used
# for the cost surface's NoData (9999 here), and that sentinel is NOT
# reliably tagged as real NoData in the output GeoTIFF header. `drains > 0`
# therefore matches almost the entire raster, not just the genuine path
# cells (since 9999 > 0 too) -- verified directly against the whitebox
# binary. Destinations were written with value 1 in targets_tif, so check
# for that exact value instead.
drains_bin = drains == 1

plt.figure(figsize=(10, 8))
plt.imshow(dem, cmap="terrain", alpha=0.5)
plt.imshow(streams_bin, cmap="Blues", alpha=0.6)
plt.imshow(drains_bin, cmap="Greens", alpha=0.8)
plt.imshow(unconnected, cmap="Reds", alpha=0.4)
plt.title("Alternate Drainage Network")
plt.axis("off")
plt.show()

print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY")

# ==========================================================
# STEP 12: RASTER → POLYGON
# ==========================================================
index_tif = os.path.join(OUTPUT_DIR, "waterlogging_index.tif")
hotspot_tif = os.path.join(OUTPUT_DIR, "hotspots.tif")

index_shp = os.path.join(OUTPUT_DIR, "waterlogging_index.shp")
hotspot_shp = os.path.join(OUTPUT_DIR, "hotspots.shp")

wbt.raster_to_vector_polygons(
    i=index_tif,
    output=index_shp
)

wbt.raster_to_vector_polygons(
    i=hotspot_tif,
    output=hotspot_shp
)

# ✅ check
check_file(index_shp, "Index polygon failed")
check_file(hotspot_shp, "Hotspot polygon failed")

print("✔ Polygons created")

# ==========================================================
# STEP 13: DRAINAGE VECTORIZATION
# ==========================================================
drain_streams_tif = os.path.join(OUTPUT_DIR, "drain_streams.tif")
drain_shp = os.path.join(OUTPUT_DIR, "alternative_drainage.shp")
flow_dir = os.path.join(OUTPUT_DIR, "flow_dir.tif")

# Save clean raster
with rasterio.open(filled) as src:
    profile = src.profile.copy()

profile.update(dtype="int32", nodata=0)

with rasterio.open(drain_streams_tif, "w", **profile) as dst:
    dst.write(drains_bin.astype("int32"), 1)

check_file(drain_streams_tif, "Drain raster creation failed")

# Vectorize
wbt.raster_streams_to_vector(
    streams=drain_streams_tif,
    d8_pntr=flow_dir,
    output=drain_shp
)

check_file(drain_shp, "Drainage vector failed")

print("✔ Drainage shapefile created")

# ==========================================================
# STEP 14: CHECK GDAL
# ==========================================================
import subprocess

def check_gdal():
    try:
        subprocess.run(["ogr2ogr", "--version"], check=True, capture_output=True)
        print("✔ GDAL available")
    except FileNotFoundError:
        raise RuntimeError(
            "❌ GDAL not installed.\n"
            "Install manually:\n"
            "sudo apt install gdal-bin\n"
            "or conda install -c conda-forge gdal"
        )

check_gdal()

# ==========================================================
# STEP 15: SHP → GPKG
# ==========================================================
gpkg_file = os.path.join(OUTPUT_DIR, "final_outputs.gpkg")

if os.path.exists(gpkg_file):
    os.remove(gpkg_file)

first = True

for file in os.listdir(OUTPUT_DIR):

    if file.endswith(".shp"):

        full_path = os.path.join(OUTPUT_DIR, file)
        layer_name = os.path.splitext(file)[0]

        cmd = [
            "ogr2ogr",
            "-f", "GPKG",
            gpkg_file,
            full_path,
            "-nln", layer_name
        ]

        if not first:
            cmd.insert(4, "-append")

        subprocess.run(cmd, check=True)

        print(f"✔ Added {file} → layer: {layer_name}")

        first = False

check_file(gpkg_file, "GPKG creation failed")

print("✔ All shapefiles merged into GPKG")

# ==========================================================
# STEP 16: CONVERT TO COG
# ==========================================================
def convert_all_to_cog(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):

        if file.endswith(".tif") and not file.endswith("_cog.tif"):

            in_path = os.path.join(input_folder, file)
            out_path = os.path.join(output_folder, file.replace(".tif", "_cog.tif"))

            if os.path.exists(out_path):
                continue

            cmd = [
                "gdal_translate",
                in_path,
                out_path,
                "-of", "COG",
                "-co", "COMPRESS=LZW"
            ]

            subprocess.run(cmd, check=True)

            print(f"✔ COG: {file}")

    print("✔ All COGs created")


convert_all_to_cog(OUTPUT_DIR, os.path.join(OUTPUT_DIR, "cog"))
