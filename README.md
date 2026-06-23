# MoPR Hackathon Repository

This repository provides a complete pipeline for processing LiDAR point cloud data, performing classification using RandLA-Net, generating Digital Terrain Models (DTMs), and conducting hydrological analysis for drainage network assessment.

---

## Overview

The workflow includes:

1. **Model Setup (RandLA-Net)**
2. **Pre-processing of Point Cloud Data**
3. **Point Cloud Classification**
4. **Post-processing and Export**
5. **DTM Generation (COG format)**
6. **Hydrological Modelling & Drainage Analysis**

---

## Installation & Build

The model setup and environment configuration should follow the official RandLA-Net PyTorch implementation:

https://github.com/idsia-robotics/RandLA-Net-pytorch

Please ensure all dependencies (PyTorch, CUDA, etc.) are properly installed.

---

## Download Pre-trained Model (Required)

Before running classification, download the pre-trained model:

https://iitk-my.sharepoint.com/:u:/g/personal/moonisali20_iitk_ac_in/IQBU27BkqBoxTZGtqK8AJ4yaAbg6nsPaUBPqKKwCjHmElNM?e=R38gZn

Steps:

1. Download the `.zip` file
2. Extract the contents
3. Place the extracted folder in:

```bash
data/saved_models/
```

---

## Dataset Structure (Important)

Each point cloud must be stored in a separate folder following a strict structure:

```
data/
│
├── pc_id=1/
│   ├── pc.pickle
│   └── metadata/
│       └── metadata.pickle
│
├── pc_id=2/
│   ├── pc.pickle
│   └── metadata/
│       └── metadata.pickle
```

### Requirements

* Each folder must be named as:
  `pc_id=<integer_id>`

* Inside each folder:

  * `pc.pickle` → contains the point cloud data
  * `metadata/metadata.pickle` → contains metadata information


---

## Workflow

### 1. Pre-processing (LAZ → PKL)

Convert raw `.laz` point cloud files into `.pkl` format required by the model:

```bash
python pc2pickle.py
```

---

### 2. Classification

Run the trained RandLA-Net model for classification:

```bash
python test.py
```

Ensure that:

* Pre-trained model is placed in `data/saved_models/`
* Dataset follows the required folder structure
* File paths are correctly specified
* Model weights are properly loaded

---

### 3. Post-processing (PKL → LAZ)

Convert classified `.pkl` outputs back to `.laz` format:

```bash
python pickle2pc.py
```

---

### Alternative: Direct Processing (Recommended)

You can skip intermediate steps and directly process `.laz` files:

```bash
python test_laz.py
```

This will:

* Perform classification
* Directly generate classified `.laz` output

---

### 4. DTM Generation (COG Format)

Generate Digital Terrain Model (DTM) in Cloud Optimized GeoTIFF (COG) format:

```bash
python las2cog.py
```

---

### 5. Hydrological Modelling

Perform hydrological analysis and drainage network extraction:

```bash
python waterlogging.py
```

This script performs:

* Depression filling
* Flow direction computation
* Flow accumulation
* Natural drainage network extraction
* Overlay of streams on identified hotspots
* Detection of unconnected hotspots
* Proposal of alternate drainage network
* Save all layers in folder "outputs"

---

### Alternative: Hydrological Modelling with Visualisation (Recommended)

You can visualise each layer before saving as output during hydrological analysis and drainage network extraction:

```bash
python waterlogging_with_visualisation.py
```

waterlogging.ipynb is also provided, which you can run in Google Colab or Jupyter Notebook to understand and visualise each layer at different steps of the hydrological analysis.

---

## Required Resources

* **Computing:** GPU-enabled system (recommended)
* **Software Stack:**

  * PyTorch
  * GDAL
  * WhiteboxTools
  * CloudCompare
  * QGIS
* **Data:** High-resolution LiDAR point cloud datasets (`.laz`)

---

## Notes

* Ensure coordinate systems are consistent across all processing steps.
* Strictly follow dataset structure to avoid runtime errors.
* Large datasets may require significant memory and processing time.

---
