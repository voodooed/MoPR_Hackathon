# ==========================================================
# WATERLOGGING + DRAINAGE PIPELINE (WITH VISUALISATION)
# ==========================================================

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from whitebox.whitebox_tools import WhiteboxTools
import subprocess

# ==========================================================
# CONFIG
# ==========================================================
INPUT_DEM = "DTM.tif"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# HELPERS
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

def show_raster(arr, title="", cmap="viridis"):
    plt.figure(figsize=(8, 6))
    vmin = np.nanpercentile(arr, 2)
    vmax = np.nanpercentile(arr, 98)
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ==========================================================
# STEP 1: FILL DEM
# ==========================================================
filled = os.path.join(OUTPUT_DIR, "dtm_filled.tif")

wbt.fill_depressions(INPUT_DEM, filled)
check_file(filled, "FillDepressions failed")

dem = load_raster(filled)
show_raster(dem, "Filled DEM", "terrain")

# ==========================================================
# STEP 2: FLOW DIR
# ==========================================================
flow_dir = os.path.join(OUTPUT_DIR, "flow_dir.tif")

wbt.d8_pointer(filled, flow_dir)
check_file(flow_dir, "Flow direction failed")

flow_dir_arr = load_raster(flow_dir)
show_raster(flow_dir_arr, "Flow Direction")

# ==========================================================
# STEP 3: FLOW ACC
# ==========================================================
flow_acc = os.path.join(OUTPUT_DIR, "flow_acc.tif")

wbt.d8_flow_accumulation(filled, flow_acc, out_type="cells")
check_file(flow_acc, "Flow accumulation failed")

flow = load_raster(flow_acc)
flow[flow <= 0] = np.nan
show_raster(np.log1p(flow), "Flow Accumulation (log)", "Blues")

# ==========================================================
# STEP 4: SLOPE
# ==========================================================
slope = os.path.join(OUTPUT_DIR, "slope.tif")

wbt.slope(filled, slope, units="degrees")
check_file(slope, "Slope failed")

slope_arr = load_raster(slope)
show_raster(slope_arr, "Slope", "inferno")

# ==========================================================
# STEP 5: TWI
# ==========================================================
twi = os.path.join(OUTPUT_DIR, "twi.tif")

wbt.wetness_index(flow_acc, slope, twi)
check_file(twi, "TWI failed")

twi_arr = load_raster(twi)
show_raster(twi_arr, "TWI", "YlGnBu")

# ==========================================================
# STEP 6: STREAMS + HAND
# ==========================================================
streams = os.path.join(OUTPUT_DIR, "streams.tif")
hand = os.path.join(OUTPUT_DIR, "hand.tif")

wbt.extract_streams(flow_acc, streams, threshold=1000)
check_file(streams, "Streams failed")

wbt.elevation_above_stream(filled, streams, hand)
check_file(hand, "HAND failed")

streams_arr = load_raster(streams)
streams_bin = streams_arr > 0
show_raster(streams_bin.astype(float), "Streams", "Blues")

hand_arr = load_raster(hand)
show_raster(hand_arr, "HAND")

# ==========================================================
# STEP 7: WATERLOGGING INDEX
# ==========================================================
twi_n = normalize(twi_arr)
slope_n = 1 - normalize(slope_arr)
elev_n = 1 - normalize(dem)
hand_n = 1 - normalize(hand_arr)

index = (twi_n + slope_n + elev_n + hand_n) / 4
index[np.isnan(dem)] = np.nan

hotspots = index >= np.nanpercentile(index, 70)

show_raster(index, "Waterlogging Index", "RdYlBu_r")

plt.imshow(index, cmap="terrain", alpha=0.7)
plt.imshow(hotspots, cmap="Reds", alpha=0.5)
plt.title("Hotspots")
plt.axis("off")
plt.show()

# SAVE
with rasterio.open(filled) as src:
    meta = src.meta.copy()

meta.update(dtype="float32", nodata=np.nan)

index_tif = os.path.join(OUTPUT_DIR, "waterlogging_index.tif")
with rasterio.open(index_tif, "w", **meta) as dst:
    dst.write(index.astype("float32"), 1)

meta.update(dtype="uint8", nodata=0)

hotspot_tif = os.path.join(OUTPUT_DIR, "hotspots.tif")
with rasterio.open(hotspot_tif, "w", **meta) as dst:
    dst.write(hotspots.astype("uint8"), 1)

# ==========================================================
# STEP 8: COST
# ==========================================================
unconnected = hotspots & (~streams_bin)

cost = 0.5*slope_n + 0.3*hand_n + 0.2*elev_n
cost[np.isnan(cost)] = 9999

show_raster(cost, "Cost Surface")

cost_tif = os.path.join(OUTPUT_DIR, "cost.tif")
targets_tif = os.path.join(OUTPUT_DIR, "targets.tif")

meta.update(dtype="float32", nodata=9999)
with rasterio.open(cost_tif, "w", **meta) as dst:
    dst.write(cost.astype("float32"), 1)

meta.update(dtype="int32", nodata=0)
with rasterio.open(targets_tif, "w", **meta) as dst:
    dst.write(unconnected.astype("int32"), 1)

# ==========================================================
# STEP 9: COST DISTANCE
# ==========================================================
backlink = os.path.join(OUTPUT_DIR, "backlink.tif")
cost_dist = os.path.join(OUTPUT_DIR, "cost_dist.tif")

wbt.cost_distance(streams, cost_tif, cost_dist, backlink)
check_file(backlink, "CostDistance failed")

# ==========================================================
# STEP 10: COST PATH
# ==========================================================
drain = os.path.join(OUTPUT_DIR, "drain_alternative.tif")

wbt.cost_pathway(targets_tif, backlink, drain)
check_file(drain, "CostPathway failed")

drains = load_raster(drain)
# FIX: cost_pathway fills every "off-path" cell with the SAME sentinel used
# for the cost surface's NoData (9999 here), and that sentinel is NOT
# reliably tagged as real NoData in the output GeoTIFF header. `drains > 0`
# therefore matches almost the entire raster, not just the genuine path
# cells (since 9999 > 0 too) -- verified directly against the whitebox
# binary. Destinations were written with value 1 in targets_tif, so check
# for that exact value instead.
drains_bin = drains == 1

plt.imshow(dem, cmap="terrain", alpha=0.5)
plt.imshow(streams_bin, cmap="Blues", alpha=0.5)
plt.imshow(drains_bin, cmap="Greens", alpha=0.8)
plt.imshow(unconnected, cmap="Reds", alpha=0.4)
plt.title("Drainage Network")
plt.axis("off")
plt.show()

# ==========================================================
# STEP 11: VECTOR OUTPUTS
# ==========================================================
index_shp = os.path.join(OUTPUT_DIR, "waterlogging_index.shp")
hotspot_shp = os.path.join(OUTPUT_DIR, "hotspots.shp")

wbt.raster_to_vector_polygons(index_tif, index_shp)
wbt.raster_to_vector_polygons(hotspot_tif, hotspot_shp)

# Drain vector
drain_streams = os.path.join(OUTPUT_DIR, "drain_streams.tif")

with rasterio.open(filled) as src:
    profile = src.profile.copy()

profile.update(dtype="int32", nodata=0)

with rasterio.open(drain_streams, "w", **profile) as dst:
    dst.write(drains_bin.astype("int32"), 1)

drain_shp = os.path.join(OUTPUT_DIR, "alternative_drainage.shp")

wbt.raster_streams_to_vector(drain_streams, flow_dir, drain_shp)

# ==========================================================
# STEP 12: GPKG
# ==========================================================
gpkg = os.path.join(OUTPUT_DIR, "final_outputs.gpkg")
if os.path.exists(gpkg):
    os.remove(gpkg)

first = True
for f in os.listdir(OUTPUT_DIR):
    if f.endswith(".shp"):
        path = os.path.join(OUTPUT_DIR, f)
        layer = os.path.splitext(f)[0]

        cmd = ["ogr2ogr", "-f", "GPKG", gpkg, path, "-nln", layer]
        if not first:
            cmd.insert(4, "-append")

        subprocess.run(cmd, check=True)
        first = False

# ==========================================================
# STEP 13: COG
# ==========================================================
cog_dir = os.path.join(OUTPUT_DIR, "cog")
os.makedirs(cog_dir, exist_ok=True)

for f in os.listdir(OUTPUT_DIR):
    if f.endswith(".tif"):
        in_path = os.path.join(OUTPUT_DIR, f)
        out_path = os.path.join(cog_dir, f.replace(".tif", "_cog.tif"))

        subprocess.run([
            "gdal_translate", in_path, out_path,
            "-of", "COG", "-co", "COMPRESS=LZW"
        ], check=True)

print("\n ALL DONE SUCCESSFULLY")
