# AI assistance: this file was written with the assistance of DeepSeek V3 (2026).
# The code was reviewed, tested, and modified by Alexander Zhang.

import rasterio
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Import paths from central configuration
from paths import (
    HUMBER_RASTER_DIR,
    DON_RASTER_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    HUMBER_TARGET_FILE
)

# ----- Load the trained Don River model -----
model_path = RESULTS_DIR / "random_forest_model.pkl"

if not model_path.exists():
    print(f"❌ Model not found at: {model_path}")
    print("   Please save your trained Don River model first.")
    exit()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print("✅ Loaded trained Don River model")

# ----- Humber features (in the SAME ORDER as Don model) -----
feature_files = [
    "basins_realign.tif",           # 1 - categorical
    "drainage_realign.tif",      # 2 - categorical
    "flowaccum_realign.tif",   # 3 - continuous
    "flowdir_realign.tif",          # 4 - categorical
    "half-basins_realign.tif",       # 5 - categorical
    "lengthsteepness_realign.tif", # 6 - continuous
    "steepness_realign.tif", # 7 - continuous
    "spi_realign.tif",         # 8 - continuous
    "streanseg_realign.tif",       # 9 - categorical
    "topographic_realign.tif",         # 10 - continuous
    "proximity_realign.tif",    # 11 - continuous
    "precip_realign.tif" # 12 - continuous
]

# ----- Load Humber features -----
print("\n📊 Loading Humber features:")
feature_arrays = []
src_ref = None

for fpath in feature_files:
    full_path = HUMBER_RASTER_DIR / fpath
    if not full_path.exists():
        print(f"   ❌ File not found: {fpath}")
        exit()
    with rasterio.open(full_path) as src:
        arr = src.read(1).astype(np.float32)
        feature_arrays.append(arr)
        if src_ref is None:
            src_ref = src
        print(f"   ✅ Loaded: {fpath}")

# Stack features
features_stack = np.stack(feature_arrays, axis=0)
height, width = features_stack.shape[1], features_stack.shape[2]
print(f"\n✅ Features stack shape: {features_stack.shape}")

# ----- Load Humber target (TRCA floodplain) -----
target_path = HUMBER_RASTER_DIR / "floodplain_raster_humber2.tif"

if not target_path.exists():
    print(f"\n❌ Target file not found: {target_path}")
    exit()

with rasterio.open(target_path) as src:
    target = src.read(1).astype(np.float32)
    meta = src.meta

print(f"✅ Target raster shape: {target.shape}")

# ----- Prepare data -----
n_features, height, width = features_stack.shape
X = features_stack.reshape(n_features, height * width).T
y = target.flatten()

nodata_value = -3.4028234663852886e+38

valid_target_mask = (y != nodata_value) & np.isfinite(y)
valid_features_mask = ~np.any(~np.isfinite(X), axis=1)
valid_mask = valid_features_mask & valid_target_mask

X_clean = X[valid_mask]
y_clean = y[valid_mask]

print(f"\n✅ Valid pixels: {len(X_clean):,}")

# ----- Make predictions -----
print("\n🔮 Making predictions on Humber River...")
y_pred = model.predict(X_clean)
y_pred_proba = model.predict_proba(X_clean)[:, 1]

# ----- Evaluate -----
accuracy = accuracy_score(y_clean, y_pred)
roc_auc = roc_auc_score(y_clean, y_pred_proba)

print("\n" + "="*50)
print("HUMBER RIVER GENERALIZATION TEST RESULTS")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_clean, y_pred, target_names=['Non-Floodplain', 'Floodplain']))

cm = confusion_matrix(y_clean, y_pred)
print("\nConfusion Matrix:")
print(f"True Negatives: {cm[0,0]:,} | False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,} | True Positives: {cm[1,1]:,}")

# ----- Compare to Don River results -----
print("\n" + "="*50)
print("COMPARISON: DON vs HUMBER")
print("="*50)
print(f"{'Metric':<20} {'Don River':<15} {'Humber River':<15} {'Drop':<15}")
print("-"*65)

don_accuracy = 0.9326
don_roc_auc = 0.9859

print(f"{'Accuracy':<20} {don_accuracy:<15.4f} {accuracy:<15.4f} {don_accuracy - accuracy:<15.4f}")
print(f"{'ROC-AUC':<20} {don_roc_auc:<15.4f} {roc_auc:<15.4f} {don_roc_auc - roc_auc:<15.4f}")

# ----- Export predictions to raster -----
print("\n💾 Exporting predictions...")

predicted_proba_full = np.full(height * width, np.nan, dtype=np.float32)
predicted_proba_full[valid_mask] = y_pred_proba
predicted_proba = predicted_proba_full.reshape(height, width)

predicted_binary_full = np.full(height * width, np.nan, dtype=np.float32)
predicted_binary_full[valid_mask] = y_pred
predicted_binary = predicted_binary_full.reshape(height, width)

out_meta = meta.copy()
out_meta.update({"dtype": "float32", "count": 1, "nodata": np.nan})

prob_path = RESULTS_DIR / "predicted_floodplain_probability_humber.tif"
with rasterio.open(prob_path, "w", **out_meta) as dst:
    dst.write(predicted_proba.astype(np.float32), 1)

binary_path = RESULTS_DIR / "predicted_floodplain_binary_humber.tif"
with rasterio.open(binary_path, "w", **out_meta) as dst:
    dst.write(predicted_binary.astype(np.float32), 1)

print(f"✅ Probability map: {prob_path}")
print(f"✅ Binary map: {binary_path}")

# ----- Comparison plot -----
binary_cmap = ListedColormap(['lightgray', 'red'])

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(target, cmap=binary_cmap, interpolation='none', vmin=0, vmax=1)
plt.title('Actual TRCA Floodplain (Humber)')
legend_elements = [
    Patch(facecolor='red', label='Floodplain (1)'),
    Patch(facecolor='lightgray', label='Non-Floodplain (0)')
]
plt.legend(handles=legend_elements, loc='lower right')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(predicted_proba, cmap='RdYlBu_r', interpolation='none', vmin=0, vmax=1)
plt.title('Predicted Floodplain Probability')
plt.colorbar(label='Probability', shrink=0.8)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(predicted_binary, cmap=binary_cmap, interpolation='none', vmin=0, vmax=1)
plt.title('Predicted Binary Floodplain')
plt.legend(handles=legend_elements, loc='lower right')
plt.axis('off')

plt.suptitle(f'Humber River: Actual vs Predicted\nAccuracy: {accuracy:.3f} | ROC-AUC: {roc_auc:.3f}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

plt.savefig(FIGURES_DIR / "humber_generalization_comparison.png", dpi=300, bbox_inches='tight')
print(f"✅ Comparison plot saved: {FIGURES_DIR / 'humber_generalization_comparison.png'}")
plt.show()

print("\n📊 DETAILED STATISTICS:")
print("-" * 40)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
