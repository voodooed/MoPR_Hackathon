## Save DEM as COG

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# ================== INPUT ==================
input_laz = "classified_pc.laz"
output_cog = "DTM.tif"
resolution = 5
nodata = -9999

# ================== READ LAZ ==================
las = laspy.read(input_laz)

x = las.x
y = las.y
z = las.z
classification = las.classification

# ================== FILTER GROUND ==================
ground_mask = (classification == 1)
x = x[ground_mask]
y = y[ground_mask]
z = z[ground_mask]

print(f"Ground points: {len(z)}")

# ================== CREATE GRID ==================
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

ncols = int((xmax - xmin) / resolution) + 1
nrows = int((ymax - ymin) / resolution) + 1

dem = np.full((nrows, ncols), np.nan, dtype=np.float32)

# ================== RASTERIZATION (MIN Z) ==================
col = ((x - xmin) / resolution).astype(int)
row = ((ymax - y) / resolution).astype(int)

for i in range(len(z)):
    r, c = row[i], col[i]
    if np.isnan(dem[r, c]) or z[i] < dem[r, c]:
        dem[r, c] = z[i]

# ================== AOI (CONVEX HULL) ==================
points_2d = np.vstack((x, y)).T
hull = ConvexHull(points_2d)
hull_points = points_2d[hull.vertices]

# Grid coordinates
grid_x, grid_y = np.meshgrid(
    np.linspace(xmin, xmax, ncols),
    np.linspace(ymax, ymin, nrows)
)

grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
hull_path = Path(hull_points)

inside_mask = hull_path.contains_points(grid_points)
inside_mask = inside_mask.reshape((nrows, ncols))

# ================== INTERPOLATION ==================
mask = np.isnan(dem)

# Nearest neighbor fill
indices = ndimage.distance_transform_edt(
    mask,
    return_distances=False,
    return_indices=True
)
filled_dem = dem[tuple(indices)]

# Optional smoothing (recommended)
filled_dem = ndimage.gaussian_filter(filled_dem, sigma=1)

# Apply AOI mask
filled_dem[~inside_mask] = nodata
filled_dem[np.isnan(filled_dem)] = nodata

# ================== SAVE AS COG (DIRECT, NO ERRORS) ==================
transform = from_origin(xmin, ymax, resolution, resolution)

with rasterio.open(
    output_cog,
    'w',
    driver='COG',
    height=filled_dem.shape[0],
    width=filled_dem.shape[1],
    count=1,
    dtype=filled_dem.dtype,
    crs="EPSG:32643",  # CHANGE THIS if needed
    transform=transform,
    nodata=nodata,
    compress="DEFLATE",
    blocksize=512
) as dst:
    dst.write(filled_dem, 1)

print(f"COG DEM saved successfully: {output_cog}")
