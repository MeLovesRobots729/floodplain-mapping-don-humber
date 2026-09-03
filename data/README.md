```markdown
# Data Folder

This folder contains the input data for the floodplain mapping project.

## ⚠️ Important

**Raster files (GeoTIFFs) are NOT included in this repository** due to file size limits. You need to download them from the sources below.

## 📁 Folder Structure

```
data/
├── don_river/          # Don River raster files (training)
├── humber_river/       # Humber River raster files (testing)
└── precipitation/      # TRCA precipitation CSV files
```

## 📥 Where to Download the Data

| Data Type | Source | Link |
|-----------|--------|------|
| Digital Elevation Model (30m) | NASADEM | https://doi.org/10.5069/G93T9FD9 |
| TRCA Floodplain Data | TRCA Open Data | https://trca-camaps.opendata.arcgis.com/datasets/33630fb21fcc400eb78eda205a90949c_0/explore?location=43.802167%2C-79.501537%2C10 |
| TRCA Precipitation Data | TRCA Open Data | https://data.trca.ca/dataset/precipitation |
| TRCA River Network | TRCA Open Data | https://trca-camaps.opendata.arcgis.com/datasets/162daaab43b744e5b80a2d77b882df8a_0/explore?location=43.660600%2C-79.220341%2C9 |
| TRCA Watershed Boundaries | TRCA Open Data | https://trca-camaps.opendata.arcgis.com/datasets/dd30e4650f38495399801deb2ea33aa6_0/explore?location=43.733964%2C-79.479766%2C11 |

## 🔧 How to Set Up

1. Download the data from the links above
2. Place Don River files in `data/don_river/`
3. Place Humber River files in `data/humber_river/`
4. Place precipitation CSVs in `data/precipitation/`
5. Run the scripts from the `code/` folder

## 📄 Files in This Folder

- `README.md` - This file
- `station_locations.csv` - TRCA gauge coordinates and mean annual precipitation
- `data_sources.txt` - Full list of data sources with links

## 📝 Required Files

### Don River (training)
- `basins_normalized.tif`
- `drainagedir_normalized.tif`
- `flowaccum_normalized.tif`
- `flowdir_normalized.tif`
- `halfbasins_normalized.tif`
- `slopelength_normalized.tif`
- `slopesteepness_normalized.tif`
- `streampowerindex_normalized.tif`
- `streamsegs_normalized.tif`
- `topographicindex_normalized.tif`
- `distance_River_aligend2.tif`
- `floodplain_realigned.tif`
- `precipitation_mean_annual.tif`

### Humber River (testing)
- `basins_humber.tif`
- `drainagedir_humber.tif`
- `flowaccum_humber_norm.tif`
- `flowdir_humber.tif`
- `halfbasins_humber.tif`
- `slopelength_humber_norm.tif`
- `slopesteepness_humber_norm.tif`
- `spi_humber_norm.tif`
- `streamsegs_humber.tif`
- `twi_humber_norm.tif`
- `distance_humber_norm.tif`
- `precipitation_humber_norm.tif`
- `floodplain_humber_realign.tif`
- `humber_dem.tif`

### Precipitation
- `station_locations.csv`
- `Precip Total.Final@HY038.csv`
- `Precip Total.Final@HY067.csv`
```