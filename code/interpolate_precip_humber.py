import pandas as pd
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from pathlib import Path
import matplotlib.pyplot as plt

# Import paths from central configuration
from paths import (
    HUMBER_RASTER_DIR,
    PRECIP_DATA_DIR,
    FIGURES_DIR
)

# Humber gauge data - REPLACE with actual coordinates from metadata
humber_stations = pd.DataFrame({
    'station_id': ['HY038', 'HY067'],
    'X': [631696, 622289],      # UTM Easting (replace with actual values)
    'Y': [4838184, 4854337],    # UTM Northing (replace with actual values)
    'mean_annual': [579.43, 524.23]  # From calculation
})

print("📍 Humber Precipitation Gauges:")
print(humber_stations)

# Load Humber DEM (must be in EPSG:32617)
dem_path = HUMBER_RASTER_DIR / "dem_aligned.tif"
if not dem_path.exists():
    dem_path = HUMBER_RASTER_DIR / "dem_humber.tif"
    if not dem_path.exists():
        print(f"❌ DEM not found! Looking for: {dem_path}")
        print("   Please update the path to the Humber DEM.")
        exit()

with rasterio.open(dem_path) as src:
    dem = src.read(1)
    transform = src.transform
    crs = src.crs
    height, width = dem.shape
    nodata = src.nodata

print(f"\n📏 DEM: {height}×{width} pixels, CRS: {crs}")

# Create grid of pixel coordinates
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
xs, ys = rasterio.transform.xy(transform, rows, cols)
xs_flat = np.array(xs).flatten()
ys_flat = np.array(ys).flatten()

# Extract station coordinates and values
station_xs = humber_stations['X'].values
station_ys = humber_stations['Y'].values
station_values = humber_stations['mean_annual'].values

# IDW Interpolation
print("\n📊 Interpolating precipitation using IDW...")

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

# Save the raster
out_meta = src.meta.copy()
out_meta.update({
    "dtype": "float32",
    "count": 1,
    "nodata": -9999
})

output_path = HUMBER_RASTER_DIR / "precipitation_humber.tif"
with rasterio.open(output_path, "w", **out_meta) as dst:
    raster_to_save = np.where(np.isnan(precip_raster), -9999, precip_raster)
    dst.write(raster_to_save.astype(np.float32), 1)

print(f"\n✅ Saved precipitation raster to: {output_path}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

im1 = ax1.imshow(precip_raster, cmap='Blues', interpolation='none')
ax1.set_title('Humber River - Mean Annual Precipitation', fontsize=12)
ax1.axis('off')
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
cbar1.set_label('mm')

ax2.imshow(precip_raster, cmap='Blues', alpha=0.5, interpolation='none')

for i, (x, y, sid, val) in enumerate(zip(station_xs, station_ys, 
                                         humber_stations['station_id'], 
                                         station_values)):
    col = int((x - transform[2]) / transform[0])
    row = int((y - transform[5]) / transform[4])
    if 0 <= row < height and 0 <= col < width:
        ax2.scatter(col, row, c='red', s=100, edgecolor='white', zorder=5)
        ax2.annotate(f"{sid}\n{val:.0f}mm", (col, row), 
                    xytext=(5, 5), textcoords='offset points',
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))

ax2.set_title('Stations Overlay', fontsize=12)
ax2.axis('off')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "precipitation_humber_interpolation.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved visualization to: {FIGURES_DIR / 'precipitation_humber_interpolation.png'}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Stations used: {len(humber_stations)}")
print(f"Mean annual precipitation range: {station_values.min():.1f} - {station_values.max():.1f} mm")
print(f"Raster shape: {precip_raster.shape}")
print("="*60)