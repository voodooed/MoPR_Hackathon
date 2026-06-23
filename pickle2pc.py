import numpy as np
import pickle
import laspy

# -------- INPUT FILES --------
xyz = pickle.load(open("data/saved_models/MoPR_whole/output/segmentations/example/xyz_tile.pickle", "rb"))        # (n,3)
labels = pickle.load(open("data/saved_models/MoPR_whole/output/segmentations/example/xyz_labels.pickle", "rb"))   # (n,)
rgb = pickle.load(open("data/saved_models/MoPR_whole/output/segmentations/example/true_rgb.pickle", "rb"))        # (n,3)

# -------- CREATE LAS HEADER --------
header = laspy.LasHeader(point_format=3, version="1.2")

# -------- CREATE LAS OBJECT --------
las = laspy.LasData(header)

# Assign coordinates
las.x = xyz[:, 0]
las.y = xyz[:, 1]
las.z = xyz[:, 2]

# Assign RGB (scale to 16-bit if needed)
las.red = (rgb[:, 0] * 256).astype(np.uint16)
las.green = (rgb[:, 1] * 256).astype(np.uint16)
las.blue = (rgb[:, 2] * 256).astype(np.uint16)

# Assign predicted labels → classification
las.classification = labels.astype(np.uint8)

# -------- SAVE AS LAZ --------
las.write("predicted_whole.laz")

print("LAZ file saved: predicted.laz")
