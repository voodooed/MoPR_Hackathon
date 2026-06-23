from model.testing import segment_randlanet
from model.hyperparameters import hyp
import pickle
import numpy as np
import laspy
import os

# ================== CONFIG ==================
MODEL_PATH = "data/saved_models/MoPR_whole/"
PC_PATH = "data/pc_id=2/"
SEG_NAME = "example"

OUTPUT_DIR = os.path.join(MODEL_PATH, "output/segmentations", SEG_NAME)
OUTPUT_LAS = os.path.join(OUTPUT_DIR, "classified_pc.laz")

# ================== STEP 1: RUN SEGMENTATION ==================
print("Running RandLA-Net segmentation...")

segment_randlanet(
    model_path=MODEL_PATH,
    pc_path=PC_PATH,
    cfg=hyp,
    num_workers=4,
    segmentation_name=SEG_NAME
)

print("Segmentation completed.")

# ================== STEP 2: LOAD GENERATED PICKLES ==================
print("Loading segmentation outputs...")

xyz = pickle.load(open(os.path.join(OUTPUT_DIR, "xyz_tile.pickle"), "rb"))
labels = pickle.load(open(os.path.join(OUTPUT_DIR, "xyz_labels.pickle"), "rb"))
rgb = pickle.load(open(os.path.join(OUTPUT_DIR, "true_rgb.pickle"), "rb"))

# ================== STEP 3: CREATE LAS ==================
print("Creating LAS/LAZ file...")

header = laspy.LasHeader(point_format=3, version="1.2")
las = laspy.LasData(header)

# Coordinates
las.x = xyz[:, 0]
las.y = xyz[:, 1]
las.z = xyz[:, 2]

# RGB (convert 0–255 → 0–65535 if needed)
if rgb.max() <= 1.0:
    rgb = rgb * 255

las.red = (rgb[:, 0] * 256).astype(np.uint16)
las.green = (rgb[:, 1] * 256).astype(np.uint16)
las.blue = (rgb[:, 2] * 256).astype(np.uint16)

# Classification labels
las.classification = labels.astype(np.uint8)

# ================== STEP 4: SAVE ==================
las.write(OUTPUT_LAS)

print(f"✅ LAZ file saved at: {OUTPUT_LAS}")
