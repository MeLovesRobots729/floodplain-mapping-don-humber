import numpy as np
import rasterio
from pathlib import Path
from scipy.spatial import cKDTree

# ----- REPLACE WITH YOUR ACTUAL COORDINATES FROM THE ATTRIBUTE TABLE -----
stations = {
    'HY038': {'x': 611464, 'y': 4862035, 'value': 579.43},  # Replace with actual values
    'HY067': {'x': 606340, 'y': 4845842, 'value': 524.23}   # Replace with actual values
}

# Pathss
humber_dir = Path("/Users/alexanderzhang/Desktop/Humber - Aligned Layers")

# Load Humber DEM
dem_path = humber_dir / "dem_aligned.tif"

with rasterio.open(dem_path) as src:
    dem = src.read(1)
    transform = src.transform
    height, width = dem.shape
    nodata = src.nodata

# Create grid
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
xs, ys = rasterio.transform.xy(transform, rows, cols)
xs_flat = np.array(xs).flatten()
ys_flat = np.array(ys).flatten()

# Station data
station_xs = np.array([stations[s]['x'] for s in stations])
station_ys = np.array([stations[s]['y'] for s in stations])
station_values = np.array([stations[s]['value'] for s in stations])

# IDW Interpolation
tree = cKDTree(np.column_stack([station_xs, station_ys]))
distances, indices = tree.query(np.column_stack([xs_flat, ys_flat]), k=2)

weights = 1 / (distances ** 2 + 1e-10)
weights_sum = np.sum(weights, axis=1)
predicted = np.sum(weights * station_values[indices], axis=1) / weights_sum

# Reshape to raster
precip_raster = predicted.reshape(height, width)

# Apply DEM NoData mask
nodata_mask = (dem == nodata) | np.isnan(dem)
precip_raster[nodata_mask] = np.nan

# Save
out_meta = src.meta.copy()
out_meta.update({"dtype": "float32", "count": 1, "nodata": -9999})

output_path = humber_dir / "precipitation_humber.tif"
with rasterio.open(output_path, "w", **out_meta) as dst:
    raster_to_save = np.where(np.isnan(precip_raster), -9999, precip_raster)
    dst.write(raster_to_save.astype(np.float32), 1)

print(f"✅ Saved to: {output_path}")