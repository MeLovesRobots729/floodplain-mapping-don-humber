# Floodplain Mapping with Random Forest

This repository contains all the code and data I used for my floodplain mapping project. I trained a Random Forest model on Toronto's Don River and tested it on the Humber River to see how well it generalizes.

---

## What This Project Does

I built a model that predicts floodplains using 12 different features from a 30m Digital Elevation Model. The features include things like distance to rivers, flow accumulation, and precipitation data. I trained the model on the Don River and then tested it on the Humber River without retraining to see if it would still work.

**Main results:**
- Don River: 98.6% ROC-AUC
- Humber River: 80.2% ROC-AUC (needed a lower threshold to work)

---

## What You Need to Run This

- Python 3.13 or newer
- QGIS 3.34
- GRASS GIS 8.3

### Python Libraries

```bash
pip install -r requirements.txt
```

---

## How the Code is Organized

```
floodplain-mapping-don-humber/
├── README.md              # This file (explains the project)
├── requirements.txt       # List of Python packages needed
├── LICENSE                # MIT License
├── .gitignore             # Files Git should ignore
│
├── code/                  # All Python scripts
│   ├── flood_ai_model_moran.py      # Main model training script
│   ├── test_humber_generalization.py # Generalization test on Humber
│   ├── calculate_humber_precip.py   # Calculate mean annual precipitation
│   ├── interpolate_precip.py        # Interpolate precipitation
│   ├── figure8_threshold_analysis.py # Create threshold analysis plot
│   └── paths.py           # Path configuration
│
├── data/                  # Input data (see data/README.md)
│   ├── don_river/         # Don River raster files
│   ├── humber_river/      # Humber River raster files
│   └── precipitation/     # Precipitation CSV files
│
├── figures/               # All 8 figures from the paper
│   ├── Fig1_StudyArea.png
│   ├── Fig2_Flowchart.png
│   ├── Fig3_FeatureImportance.png
│   ├── Fig4_MoranScatter.png
│   ├── Fig5_Variogram.png
│   ├── Fig6_DonPredictions.png
│   ├── Fig7_HumberPredictions.png
│   └── Fig8_ThresholdAnalysis.png
│
└── results/               # Performance metrics and outputs
    ├── README.md          # Explains what the results mean
    ├── random_forest_model.pkl      # Trained model
    ├── model_performance_summary.csv
    ├── feature_importance.csv
    └── threshold_analysis.csv
```

---

## How to Run It

### 1. Train the Don River Model

```bash
python code/flood_ai_model_moran.py
```

This script:
- Loads all 12 hydrological features
- Trains a Random Forest model on the Don River
- Calculates feature importance
- Runs spatial autocorrelation analysis
- Performs spatial block cross-validation
- Saves the trained model

### 2. Test on the Humber River

```bash
python code/test_humber_generalization.py
```

This script:
- Loads the trained Don River model
- Applies it to the Humber River without retraining
- Evaluates performance
- Finds the optimal threshold
- Saves predictions as GeoTIFFs

### 3. Run the Threshold Analysis

```bash
python code/figure8_threshold_analysis.py
```

This script:
- Creates the threshold vs. F1 score plot
- Identifies the optimal threshold (0.12)
- Saves Figure 8

### 4. Calculate Precipitation

```bash
python code/calculate_humber_precip.py
```

This script:
- Loads TRCA precipitation CSV files
- Calculates mean annual precipitation
- Prints summary of results

### 5. Interpolate Precipitation

```bash
python code/interpolate_precip.py
```

This script:
- Loads Humber DEM
- Interpolates precipitation using IDW
- Saves precipitation raster

---

## Results Summary

| Metric | Don River (Training) | Humber River (Testing) |
|--------|----------------------|------------------------|
| ROC-AUC | 0.986 | 0.802 |
| Accuracy | 93.3% | 87.5% |
| Floodplain Recall | 95.0% | 50.9% |
| Floodplain Precision | 50.0% | 26.1% |
| F1 Score | 0.61 | 0.345 |
| Optimal Threshold | 0.5 | 0.12 |

---

## Data Sources

| Data Type | Source | Link |
|-----------|--------|------|
| Digital Elevation Model (30m) | NASADEM | https://doi.org/10.5069/G93T9FD9 |
| Floodplain Data | TRCA Open Data | https://trca-camaps.opendata.arcgis.com/datasets/33630fb21fcc400eb78eda205a90949c_0/explore?location=43.802167%2C-79.501537%2C10 |
| Precipitation Data | TRCA Open Data | https://data.trca.ca/dataset/precipitation |
| River Network | TRCA Open Data | https://trca-camaps.opendata.arcgis.com/datasets/162daaab43b744e5b80a2d77b882df8a_0/explore?location=43.660600%2C-79.220341%2C9 |
| Watershed Boundaries | TRCA Open Data | https://trca-camaps.opendata.arcgis.com/datasets/dd30e4650f38495399801deb2ea33aa6_0/explore?location=43.733964%2C-79.479766%2C11 |

> **Note:** Full raster datasets (GeoTIFFs) are not included in this repository due to file size limits. You can download the data from the sources listed above.

---

Manuscript submitted to the International Journal of High School Research

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Questions or Issues?

If you have any questions or run into problems, feel free to open an issue on GitHub or contact me.

## AI Assistance
Parts of the Python code in this repository were written with the assistance of DeepSeek V3. All code was reviewed, tested, and modified by the author. Files containing AI-generated code are marked with a comment at the top.
