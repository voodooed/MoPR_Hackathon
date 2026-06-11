# SMART-DRAIN Web Application

Upload a `.las`/`.laz` point cloud → get drainage maps on OpenStreetMap.

## Quick start

```bash
# 1. Clone your pipeline repo (if not already)
git clone https://github.com/moonis-ali/MoPR_Hackathon
cd MoPR_Hackathon

# 2. Copy the two web app files into the repo root
#    app.py       ← Flask backend
#    index.html   ← Frontend (served by Flask)
#    requirements.txt

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional but recommended) Place pre-trained model
#    Download from SharePoint and extract into:
#    data/saved_models/MoPR_whole/

# 5. Run
python app.py

# 6. Open browser
#    http://localhost:5000
```

## What happens when you upload a file

| Step | What runs | Output |
|------|-----------|--------|
| 1 | laspy reads the .las/.laz | point count, CRS |
| 2 | RandLA-Net (model/testing.py) or passthrough | classified_pc.laz |
| 3 | las2cog logic — grid + fill + COG | DTM.tif |
| 4 | WhiteboxTools hydrology pipeline | flow_dir, flow_acc, TWI, HAND, streams, hotspots, drainage |
| 5 | fiona+pyproj → WGS84 GeoJSON | drainage.geojson, hotspots.geojson, alt_drainage.geojson |
| 6 | ZIP all outputs | SMART_DRAIN_outputs.zip |

## Fallback behaviour

- **No model weights**: uses existing classification field in the .las file (or marks all as ground)  
- **No fiona/shapely**: generates synthetic GeoJSON from DTM bounding box for demo purposes  
- All fallbacks are logged in the browser console panel

## Output files (in the ZIP)

- `classified_pc.laz` — RandLA-Net classified point cloud  
- `DTM.tif` — Digital Terrain Model (COG)  
- `dtm_filled.tif` — Depression-filled DTM  
- `flow_dir.tif`, `flow_acc.tif` — D8 flow direction / accumulation  
- `slope.tif`, `twi.tif`, `hand.tif` — Slope, TWI, HAND indices  
- `streams.tif`, `hotspots.tif` — Stream network, waterlogging hotspots  
- `waterlogging_index.tif` — Composite risk index  
- `drain_alternative.tif` — Proposed alternative drainage  
- `*.shp` — Vector shapefiles for all above  
- `drainage.geojson` — Natural drainage (WGS84)  
- `hotspots.geojson` — Waterlogging zones (WGS84)  
- `alt_drainage.geojson` — Proposed drain paths (WGS84)  
