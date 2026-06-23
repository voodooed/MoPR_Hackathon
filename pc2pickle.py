import laspy
import numpy as np
import pickle
import os

# -------- INPUT --------
laz_file = "KHAPRETA.laz"
pc_id = 5
output_root = "data"

# -------- READ LAZ --------
las = laspy.read(laz_file)

x = las.x
y = las.y
z = las.z

# Handle RGB (if exists)
if hasattr(las, "red"):
    r = las.red / 256.0
    g = las.green / 256.0
    b = las.blue / 256.0
else:
    r = np.zeros_like(x)
    g = np.zeros_like(x)
    b = np.zeros_like(x)


labels = las.classification


# -------- STACK --------
pc = np.vstack((x, y, z, r, g, b, labels)).T.astype(np.float32)

# -------- CREATE FOLDER --------
pc_folder = os.path.join(output_root, f"pc_id={pc_id}")
meta_folder = os.path.join(pc_folder, "metadata")

os.makedirs(meta_folder, exist_ok=True)

# -------- SAVE pc.pickle --------
with open(os.path.join(pc_folder, "pc.pickle"), "wb") as f:
    pickle.dump(pc, f)

# -------- SAVE metadata --------
metadata = {
    "pc_id": pc_id,
    "labels": list(np.unique(labels).astype(float)),
    "name": None
}

with open(os.path.join(meta_folder, "metadata.pickle"), "wb") as f:
    pickle.dump(metadata, f)

print("Conversion complete!")
print("Shape:", pc.shape)
