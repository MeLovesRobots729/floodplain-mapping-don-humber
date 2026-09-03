from pathlib import Path

# Get the base directory (go up one level from code/ to the repo root)
BASE_DIR = Path(__file__).parent.parent

# Define all your data and results paths
DON_RASTER_DIR = BASE_DIR / "data" / "don_river"
HUMBER_RASTER_DIR = BASE_DIR / "data" / "humber_river"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
PLOTS_DIR = BASE_DIR / "figures"
VARIAGRAM_DIR = BASE_DIR / "figures"

# TRCA precipitation data (where you downloaded the CSV files)
PRECIP_DATA_DIR = BASE_DIR / "data" / "precipitation"

# If you need specific files:
DON_TARGET_FILE = DON_RASTER_DIR / "floodplain_realigned.tif"
DON_DEM_PATH = DON_RASTER_DIR / "depressionlessdem_normalized.tif"
HUMBER_TARGET_FILE = HUMBER_RASTER_DIR / "floodplain_humber_realign.tif"
HUMBER_DEM_PATH = HUMBER_RASTER_DIR / "humber_dem.tif"