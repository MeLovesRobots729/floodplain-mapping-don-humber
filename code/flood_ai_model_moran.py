# AI assistance: this file was written with the assistance of DeepSeek V3 (2026).
# The code was reviewed, tested, and modified by Alexander Zhang.

import rasterio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import warnings
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path
import libpysal as lps
from esda.moran import Moran
import geopandas as gpd
from shapely.geometry import Point
import pickle

# Import paths from the central configuration file
from paths import (
    DON_RASTER_DIR,
    DON_TARGET_FILE,
    PRECIP_DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    PLOTS_DIR,
    VARIAGRAM_DIR
)

warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# 1️⃣ Load all normalized raster layers (features + target)
# ----------------------------------------------------------

raster_dir = DON_RASTER_DIR
target_file = DON_TARGET_FILE
precip_dir = PRECIP_DATA_DIR

# ---- Original feature rasters ----
raster_files = [
    raster_dir / "basins_normalized.tif",
    raster_dir / "drainagedir_normalized.tif",
    raster_dir / "flowaccum_normalized.tif",
    raster_dir / "flowdir_normalized.tif",
    raster_dir / "halfbasins_normalized.tif",
    raster_dir / "slopelength_normalized.tif",
    raster_dir / "slopesteepness_normalized.tif",
    raster_dir / "streampowerindex_normalized.tif",
    raster_dir / "streamsegs_normalized.tif",
    raster_dir / "topographicindex_normalized.tif",
    raster_dir / "distance_River_aligend2.tif"
]

target_file = raster_dir / "floodplain_realigned.tif"

# ---- Load precipitation raster (NEW) ----
precip_path = raster_dir / "precipitation_mean_annual.tif"

# Load all raster feature layers
feature_arrays = []
feature_names = []
src_ref = None

print("Loading feature rasters:")
for fpath in raster_files:
    with rasterio.open(fpath) as src:
        arr = src.read(1).astype(np.float32)
        feature_arrays.append(arr)
        name = fpath.stem
        feature_names.append(name)
        if src_ref is None:
            src_ref = src
        print(f"  ✅ Loaded: {name}")

# ---- Load and add precipitation raster ----
print("\nLoading precipitation raster...")
with rasterio.open(precip_path) as src:
    precip_arr = src.read(1).astype(np.float32)
    precip_arr[precip_arr == -9999] = np.nan
    precip_valid = precip_arr[~np.isnan(precip_arr)]
    if len(precip_valid) > 0:
        precip_min = np.nanmin(precip_valid)
        precip_max = np.nanmax(precip_valid)
        precip_norm = (precip_arr - precip_min) / (precip_max - precip_min)
        precip_norm[np.isnan(precip_arr)] = np.nan
    else:
        raise ValueError("No valid precipitation data found.")

feature_arrays.append(precip_norm)
feature_names.append("precipitation_mean_annual")
print(f"  ✅ Loaded: precipitation_mean_annual (min={precip_min:.1f}, max={precip_max:.1f})")

# Build the feature stack
features_stack = np.stack(feature_arrays, axis=0)
print(f"\nFeatures stack shape: {features_stack.shape} ({len(feature_names)} layers, {features_stack.shape[1]} rows, {features_stack.shape[2]} cols)")

# Load target raster (TRCA floodplain)
with rasterio.open(target_file) as src:
    target = src.read(1).astype(np.float32)
    meta = src.meta
print(f"Target raster shape: {target.shape}")

# ----------------------------------------------------------
# 2️⃣ Prepare data for machine learning
# ----------------------------------------------------------

n_features, height, width = features_stack.shape
X = features_stack.reshape(n_features, height * width).T
y = target.flatten()

print(f"\nOriginal data shape: {X.shape} (pixels, features)")

nodata_value = -3.4028234663852886e+38

valid_target_mask = (y != nodata_value) & np.isfinite(y)
valid_features_mask = ~np.any(~np.isfinite(X), axis=1)
valid_mask = valid_features_mask & valid_target_mask

X_clean = X[valid_mask]
y_clean = y[valid_mask]

print(f"After cleaning: {X_clean.shape} - removed {len(X) - len(X_clean)} invalid pixels")

unique_clean, counts_clean = np.unique(y_clean, return_counts=True)
print("\n🎯 CLEANED TARGET DISTRIBUTION:")
for val, count in zip(unique_clean, counts_clean):
    print(f"  Class {val}: {count} pixels ({count/len(y_clean)*100:.2f}%)")

if set(unique_clean) != {0, 1}:
    y_clean = np.where(y_clean > 0.5, 1, 0).astype(int)

# ----------------------------------------------------------
# 3️⃣ Handle drainage direction
# ----------------------------------------------------------

drainage_index = feature_names.index("drainagedir_normalized")
drainage_col = X_clean[:, drainage_index]
drainage_int = np.round(drainage_col).astype(int)
le = LabelEncoder()
drainage_encoded = le.fit_transform(drainage_int)
X_final = X_clean.copy()
X_final[:, drainage_index] = drainage_encoded
feature_names_updated = feature_names.copy()

# ----------------------------------------------------------
# 4️⃣ Train-test split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_clean, test_size=0.2, random_state=42, shuffle=True, stratify=y_clean
)

# ----------------------------------------------------------
# 5️⃣ Train model
# ----------------------------------------------------------

classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=y_clean)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1,
    max_depth=15, min_samples_split=10, min_samples_leaf=5,
    max_features='sqrt', class_weight=class_weight_dict
)

print("\nTraining Random Forest Classifier...")
model.fit(X_train, y_train)
print("✅ Model training complete!")

# ----------------------------------------------------------
# 6️⃣ Evaluate model
# ----------------------------------------------------------

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("TRCA FLOODPLAIN PREDICTION MODEL RESULTS")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Floodplain', 'Floodplain']))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"True Negatives: {cm[0,0]:,} | False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,} | True Positives: {cm[1,1]:,}")

# ----------------------------------------------------------
# 7️⃣ Feature Importance
# ----------------------------------------------------------

importances = model.feature_importances_
feature_imp_df = pd.DataFrame({
    'feature': feature_names_updated,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n📊 FEATURE IMPORTANCE RANKING:")
for idx, row in feature_imp_df.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# ----------------------------------------------------------
# 8️⃣ TASK 1.1: SPATIAL AUTOCORRELATION ANALYSIS
# ----------------------------------------------------------

print("\n" + "="*60)
print("TASK 1.1: SPATIAL AUTOCORRELATION ANALYSIS")
print("="*60)

top_features = feature_imp_df[feature_imp_df['importance'] > 0.01]['feature'].head(5).tolist()
print(f"\n📌 Analyzing top {len(top_features)} features:")
for f in top_features:
    print(f"   - {f}")

# ---------- Compute 2D valid mask for sampling ----------
nodata_value = -3.4028234663852886e+38

target_valid_2d = (target != nodata_value) & np.isfinite(target)
features_valid_2d = np.all(features_stack != nodata_value, axis=0) & np.all(np.isfinite(features_stack), axis=0)
valid_mask_2d = target_valid_2d & features_valid_2d

valid_rows, valid_cols = np.where(valid_mask_2d)
total_valid = len(valid_rows)
print(f"\n✅ Total valid pixels (2D): {total_valid}")

sample_size = min(5000, total_valid)
np.random.seed(42)
sample_indices = np.random.choice(total_valid, sample_size, replace=False)

transform = src_ref.transform
crs = src_ref.crs

moran_results = {}

# Create folder for plots (using relative paths)
plots_dir = PLOTS_DIR / "moran_plots"
plots_dir.mkdir(parents=True, exist_ok=True)
print(f"\n📁 Saving plots to: {plots_dir}")

variogram_dir = PLOTS_DIR / "variogram_plots"
variogram_dir.mkdir(parents=True, exist_ok=True)
print(f"📁 Variogram plots saved to: {variogram_dir}")

for feature_name in top_features:
    print(f"\n📍 Analyzing: {feature_name}")
    
    idx = feature_names.index(feature_name)
    raster_data = feature_arrays[idx]
    
    sample_values = []
    sample_geometries = []
    
    for idx_sample in sample_indices:
        row = valid_rows[idx_sample]
        col = valid_cols[idx_sample]
        val = raster_data[row, col]
        
        if np.isfinite(val) and val != nodata_value:
            sample_values.append(float(val))
            x, y = src_ref.xy(row, col)
            sample_geometries.append(Point(x, y))
    
    if len(sample_values) < 10:
        print(f"   ⚠️ Too few valid samples ({len(sample_values)}), skipping")
        continue
    
    gdf = gpd.GeoDataFrame(
        {'value': sample_values}, 
        geometry=sample_geometries,
        crs=crs
    )
    
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    w = lps.weights.KNN.from_array(coords, k=8)
    w.transform = 'r'
    
    moran = Moran(gdf['value'], w)
    
    moran_results[feature_name] = {
        'I': moran.I,
        'EI': moran.EI,
        'p_sim': moran.p_sim,
        'z_sim': moran.z_sim,
        'n': len(sample_values),
        'moran_object': moran,
        'gdf': gdf,
        'w': w
    }
    
    print(f"   Moran's I: {moran.I:.4f}")
    print(f"   Expected I: {moran.EI:.4f}")
    print(f"   p-value: {moran.p_sim:.4f}")
    print(f"   z-score: {moran.z_sim:.4f}")
    print(f"   Sample size: {len(sample_values)}")
    
    # ---------- CREATE MORAN SCATTER PLOT ----------
    lag = lps.weights.lag_spatial(w, gdf['value'].values)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(gdf['value'], lag, alpha=0.6, s=10, c='blue')
    
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(gdf['value'], lag)
    x_vals = np.linspace(gdf['value'].min(), gdf['value'].max(), 100)
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, 'r-', linewidth=2, label=f'Slope = {slope:.4f}')
    
    ax.plot([gdf['value'].min(), gdf['value'].max()], 
            [gdf['value'].min(), gdf['value'].max()], 
            'k--', alpha=0.5, label='y = x')
    
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Spatial Lag (Average of Neighbors)')
    ax.set_title(f"{feature_name}\nMoran's I = {moran.I:.4f}, p = {moran.p_sim:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = plots_dir / f"moran_scatter_{feature_name}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"      ✅ Saved scatter plot: {save_path}")

# ---------- CREATE LISA CLUSTER MAPS ----------
print("\n📍 Creating LISA cluster maps...")

for feature_name, results in moran_results.items():
    gdf = results['gdf']
    w = results['w']
    
    from esda.moran import Moran_Local
    lisa = Moran_Local(gdf['value'], w)
    
    gdf['lisa_cluster'] = lisa.q
    gdf['lisa_significant'] = lisa.p_sim < 0.05
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cluster_colors = {
        1: 'red',
        2: 'blue',
        3: 'lightblue',
        4: 'orange'
    }
    cluster_labels = {
        1: 'High-High',
        2: 'Low-High',
        3: 'Low-Low',
        4: 'High-Low'
    }
    
    significant = gdf[gdf['lisa_significant']]
    for cluster_type in [1, 2, 3, 4]:
        subset = significant[significant['lisa_cluster'] == cluster_type]
        if not subset.empty:
            ax.scatter(subset.geometry.x, subset.geometry.y, 
                      c=cluster_colors[cluster_type], 
                      label=cluster_labels[cluster_type],
                      s=10, alpha=0.7)
    
    non_sig = gdf[~gdf['lisa_significant']]
    if not non_sig.empty:
        ax.scatter(non_sig.geometry.x, non_sig.geometry.y, 
                  c='lightgray', label='Not Significant', s=5, alpha=0.5)
    
    ax.set_title(f"{feature_name}\nLISA Cluster Map (Local Moran's I)", fontsize=14)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(loc='lower right', fontsize=10, markerscale=2)
    
    save_path = plots_dir / f"lisa_cluster_{feature_name}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"      ✅ Saved LISA map: {save_path}")

# ---------- CREATE COMBINED SUMMARY PLOT ----------
print("📍 Creating combined summary plot...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

top_3_features = list(moran_results.keys())[:3]

for i, feature_name in enumerate(top_3_features):
    if i >= len(axes):
        break
    
    results = moran_results[feature_name]
    gdf = results['gdf']
    w = results['w']
    
    from esda.moran import Moran_Local
    lisa = Moran_Local(gdf['value'], w)
    gdf['lisa_cluster'] = lisa.q
    gdf['lisa_significant'] = lisa.p_sim < 0.05
    
    ax = axes[i]
    
    cluster_colors = {1: 'red', 2: 'blue', 3: 'lightblue', 4: 'orange'}
    significant = gdf[gdf['lisa_significant']]
    
    for cluster_type in [1, 2, 3, 4]:
        subset = significant[significant['lisa_cluster'] == cluster_type]
        if not subset.empty:
            ax.scatter(subset.geometry.x, subset.geometry.y, 
                      c=cluster_colors[cluster_type], 
                      label={1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}[cluster_type],
                      s=8, alpha=0.7)
    
    non_sig = gdf[~gdf['lisa_significant']]
    if not non_sig.empty:
        ax.scatter(non_sig.geometry.x, non_sig.geometry.y, 
                  c='lightgray', label='Not Sig', s=5, alpha=0.5)
    
    ax.set_title(f"{feature_name}\nI = {results['I']:.4f}, p = {results['p_sim']:.4f}", fontsize=11)
    ax.legend(loc='lower right', fontsize=8)

for j in range(len(top_3_features), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Local Moran's I Cluster Maps (Top Features)", fontsize=16, fontweight='bold')
plt.tight_layout()

save_path = plots_dir / "lisa_cluster_combined.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"      ✅ Saved combined plot: {save_path}")

# Summary
print("\n" + "="*60)
print("SUMMARY: GLOBAL MORAN'S I RESULTS")
print("="*60)
print(f"{'Feature':<40} {'Moran\'s I':<12} {'p-value':<12} {'Interpretation'}")
print("-"*80)

for name, results in moran_results.items():
    i_val = results['I']
    p_val = results['p_sim']
    
    if p_val < 0.05:
        if i_val > 0.7:
            interp = "Strong clustering"
        elif i_val > 0.3:
            interp = "Moderate clustering"
        else:
            interp = "Weak clustering"
    else:
        interp = "Random (not significant)"
    
    print(f"{name:<40} {i_val:<12.4f} {p_val:<12.4f} {interp}")

print("\n" + "="*60)
print("💡 INTERPRETATION:")
print("If Moran's I > 0.3 and p < 0.05 → significant spatial autocorrelation exists.")
print("This confirms that traditional random train-test split is flawed.")
print("→ Proceed to Task 1.2: Variogram analysis to find distance R")
print("="*60)

# ----------------------------------------------------------
# TASK 1.2: VARIOGRAM ANALYSIS (CLIPPED TO 5000 m)
# ----------------------------------------------------------

print("\n" + "="*60)
print("TASK 1.2: VARIOGRAM ANALYSIS (CLIPPED TO 5000 m)")
print("="*60)

try:
    from scipy.spatial.distance import pdist
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    print("✅ scipy and sklearn available")
except ImportError:
    print("⚠️ scipy not found. Installing via pip...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'scipy'])
    from scipy.spatial.distance import pdist
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score

def spherical_model(h, nugget, sill, range_):
    """Spherical variogram model"""
    h = np.array(h, dtype=np.float64)
    gamma = np.zeros_like(h)
    mask = h <= range_
    gamma[mask] = nugget + sill * (1.5 * (h[mask] / range_) - 0.5 * (h[mask] / range_) ** 3)
    gamma[~mask] = nugget + sill
    return gamma

clip_distance = 5000

variogram_features = ['distance_River_aligend2', 'halfbasins_normalized', 'basins_normalized']
variogram_results = {}

for feature_name in variogram_features:
    print(f"\n📍 Variogram for: {feature_name}")
    
    idx = feature_names.index(feature_name)
    raster_data = feature_arrays[idx]
    
    variogram_sample_size = min(15000, total_valid)
    np.random.seed(123)
    variogram_indices = np.random.choice(total_valid, variogram_sample_size, replace=False)
    
    sample_values = []
    sample_coords = []
    
    for idx_sample in variogram_indices:
        row = valid_rows[idx_sample]
        col = valid_cols[idx_sample]
        val = raster_data[row, col]
        
        if np.isfinite(val) and val != nodata_value:
            sample_values.append(float(val))
            x, y = src_ref.xy(row, col)
            sample_coords.append((x, y))
    
    if len(sample_values) < 50:
        print(f"   ⚠️ Too few samples ({len(sample_values)}), skipping")
        continue
    
    print(f"   Collected {len(sample_values)} samples")
    
    coords_array = np.array(sample_coords)
    values_array = np.array(sample_values)
    
    distances = pdist(coords_array)
    value_diffs = pdist(values_array.reshape(-1, 1), metric='sqeuclidean')
    
    valid_pairs = distances <= clip_distance
    distances_clipped = distances[valid_pairs]
    value_diffs_clipped = value_diffs[valid_pairs]
    
    if len(distances_clipped) < 100:
        print(f"   ⚠️ Too few pairs within {clip_distance}m ({len(distances_clipped)}), skipping")
        continue
    
    print(f"   Kept {len(distances_clipped)} pairs within {clip_distance}m")
    
    max_dist = clip_distance
    n_bins = 30
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    variogram_means = []
    variogram_counts = []
    
    for i in range(n_bins):
        mask = (distances_clipped >= bin_edges[i]) & (distances_clipped < bin_edges[i+1])
        if np.sum(mask) > 10:
            variogram_means.append(np.mean(value_diffs_clipped[mask]) / 2)
            variogram_counts.append(np.sum(mask))
        else:
            variogram_means.append(np.nan)
            variogram_counts.append(0)
    
    valid_bins = ~np.isnan(variogram_means)
    bin_centers_valid = bin_centers[valid_bins]
    variogram_means_valid = np.array(variogram_means)[valid_bins]
    variogram_counts_valid = np.array(variogram_counts)[valid_bins]
    
    if len(bin_centers_valid) < 5:
        print(f"   ⚠️ Too few valid bins ({len(bin_centers_valid)}), skipping")
        continue
    
    print(f"   Using {len(bin_centers_valid)} bins for fitting (0–{clip_distance}m)")
    
    try:
        p0 = [0.1, 0.5, clip_distance / 3]
        bounds = ([0, 0, 0], [1, 2, clip_distance])
        
        popt, pcov = curve_fit(
            spherical_model, 
            bin_centers_valid, 
            variogram_means_valid,
            p0=p0,
            bounds=bounds,
            maxfev=5000
        )
        
        nugget, sill, R = popt
        
        predicted = spherical_model(bin_centers_valid, *popt)
        r2 = r2_score(variogram_means_valid, predicted)
        
        variogram_results[feature_name] = {
            'nugget': nugget,
            'sill': sill,
            'range_R': R,
            'r2': r2,
            'n_samples': len(sample_values)
        }
        
        print(f"   ✅ Nugget: {nugget:.4f}")
        print(f"   ✅ Sill: {sill:.4f}")
        print(f"   ✅ Range (R): {R:.2f} meters (computed from clipped data)")
        print(f"   ✅ R²: {r2:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(bin_centers_valid, variogram_means_valid, 
                  c='blue', alpha=0.6, label='Empirical variogram')
        
        x_fit = np.linspace(0, clip_distance, 100)
        y_fit = spherical_model(x_fit, *popt)
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Spherical model (R={R:.0f}m)')
        
        ax.axvline(x=R, color='green', linestyle='--', alpha=0.7, 
                  label=f'Range (R) = {R:.0f}m')
        
        ax.set_xlabel('Distance (meters)')
        ax.set_ylabel('Semivariance')
        ax.set_title(f"Variogram: {feature_name} (0–{clip_distance}m)\nR² = {r2:.4f}, Range = {R:.0f}m")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, clip_distance)
        
        plot_path = variogram_dir / f"variogram_{feature_name}_clipped_{clip_distance}m.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"      ✅ Saved variogram plot: {plot_path}")
        
    except Exception as e:
        print(f"   ⚠️ Variogram fitting failed: {e}")
        continue

# Summary of R values
print("\n" + "="*60)
print("SUMMARY: VARIOGRAM RANGE (R) VALUES (CLIPPED TO 5000 m)")
print("="*60)
print(f"{'Feature':<40} {'Range (R) (m)':<15} {'R²':<10}")
print("-"*70)

r_values = []
for name, results in variogram_results.items():
    R_val = results['range_R']
    r2_val = results['r2']
    print(f"{name:<40} {R_val:<15.2f} {r2_val:<10.4f}")
    r_values.append(R_val)

if r_values:
    avg_R = np.mean(r_values)
    median_R = np.median(r_values)
    std_R = np.std(r_values)
    print("-"*70)
    print(f"{'AVERAGE R:':<40} {avg_R:<15.2f}")
    print(f"{'MEDIAN R:':<40} {median_R:<15.2f}")
    print(f"{'STD DEV R:':<40} {std_R:<15.2f}")
    
    recommended_R = 5000
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATION FOR SPATIAL BLOCK SIZE:")
    print(f"Use R = {recommended_R:.0f} meters for spatial block cross-validation")
    print("\n→ Proceed to Task 2.1: Standard baseline model with random split")
    print("→ Then Task 2.2: Spatial block cross-validation with R = {:.0f}m".format(recommended_R))
    print("="*60)
else:
    print("\n⚠️ No variogram results obtained. Using default R = 500 meters")
    recommended_R = 500
    print(f"→ Proceed with R = {recommended_R} meters for spatial block cross-validation")

# ----------------------------------------------------------
# TASK 2.2: SPATIAL BLOCK CROSS-VALIDATION
# ----------------------------------------------------------

print("\n" + "="*60)
print("TASK 2.2: SPATIAL BLOCK CROSS-VALIDATION")
print("="*60)

from rasterio.transform import array_bounds

bounds = array_bounds(height, width, transform)
xmin, ymin, xmax, ymax = bounds

study_width = xmax - xmin
study_height = ymax - ymin

print(f"Study area bounds: x=[{xmin:.0f}, {xmax:.0f}], y=[{ymin:.0f}, {ymax:.0f}]")
print(f"Study area dimensions: {study_width:.0f} m × {study_height:.0f} m")
print(f"Using R = {recommended_R:.0f} m for spatial blocks")

R_actual = recommended_R

n_x_blocks = max(2, int(np.ceil(study_width / R_actual)))
n_y_blocks = max(2, int(np.ceil(study_height / R_actual)))
R_actual = min(study_width / n_x_blocks, study_height / n_y_blocks)

print(f"\nUsing spatial block size: R = {R_actual:.0f} meters")
print(f"Grid: {n_x_blocks} × {n_y_blocks} = {n_x_blocks * n_y_blocks} blocks")

block_ids = np.full((height, width), -1, dtype=np.int32)

block_width_px = int(np.ceil(width / n_x_blocks))
block_height_px = int(np.ceil(height / n_y_blocks))

for i in range(n_x_blocks):
    for j in range(n_y_blocks):
        block_id = i + j * n_x_blocks
        
        col_start = i * block_width_px
        col_end = min((i + 1) * block_width_px, width)
        row_start = j * block_height_px
        row_end = min((j + 1) * block_height_px, height)
        
        block_ids[row_start:row_end, col_start:col_end] = block_id

unique_block_ids = np.unique(block_ids)
print(f"Total blocks created: {len(unique_block_ids)}")

valid_block_ids = np.unique(block_ids[valid_mask_2d])
valid_block_ids = valid_block_ids[valid_block_ids >= 0]
n_valid_blocks = len(valid_block_ids)
print(f"Valid blocks: {n_valid_blocks}")

if n_valid_blocks < 2:
    print("⚠️ Too few valid blocks. Using random split as fallback.")
    accuracy_spatial = accuracy
    roc_auc_spatial = roc_auc
    y_pred_spatial = y_pred
    y_pred_proba_spatial = y_pred_proba
    cm_spatial = cm
else:
    np.random.seed(42)
    block_ids_shuffled = np.random.permutation(valid_block_ids)
    n_train_blocks = int(0.8 * len(block_ids_shuffled))
    
    train_blocks = block_ids_shuffled[:n_train_blocks]
    test_blocks = block_ids_shuffled[n_train_blocks:]
    
    print(f"Training blocks: {len(train_blocks)}, Testing blocks: {len(test_blocks)}")
    
    train_mask = np.isin(block_ids, train_blocks)
    test_mask = np.isin(block_ids, test_blocks)
    
    train_mask = train_mask & valid_mask_2d
    test_mask = test_mask & valid_mask_2d
    
    train_pixel_count = np.sum(train_mask)
    test_pixel_count = np.sum(test_mask)
    print(f"Training pixels: {train_pixel_count:,}")
    print(f"Testing pixels: {test_pixel_count:,}")
    
    if train_pixel_count == 0 or test_pixel_count == 0:
        print("⚠️ No pixels in training or testing set. Using random split as fallback.")
        accuracy_spatial = accuracy
        roc_auc_spatial = roc_auc
        y_pred_spatial = y_pred
        y_pred_proba_spatial = y_pred_proba
        cm_spatial = cm
    else:
        train_mask_flat = train_mask.flatten()
        test_mask_flat = test_mask.flatten()
        
        y_flat = target.flatten()
        
        X_train_spatial = X[train_mask_flat]
        y_train_spatial = y_flat[train_mask_flat]
        X_test_spatial = X[test_mask_flat]
        y_test_spatial = y_flat[test_mask_flat]
        
        print(f"X_train_spatial shape: {X_train_spatial.shape}")
        print(f"X_test_spatial shape: {X_test_spatial.shape}")
        
        if len(X_train_spatial) > 0 and len(X_test_spatial) > 0:
            drainage_col_train = X_train_spatial[:, drainage_index]
            drainage_int_train = np.round(drainage_col_train).astype(int)
            drainage_encoded_train = le.transform(drainage_int_train)
            X_train_spatial[:, drainage_index] = drainage_encoded_train
            
            drainage_col_test = X_test_spatial[:, drainage_index]
            drainage_int_test = np.round(drainage_col_test).astype(int)
            drainage_encoded_test = le.transform(drainage_int_test)
            X_test_spatial[:, drainage_index] = drainage_encoded_test
            
            print("\nTraining Random Forest with SPATIAL block split...")
            model_spatial = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
                max_depth=15, min_samples_split=10, min_samples_leaf=5,
                max_features='sqrt', class_weight=class_weight_dict
            )
            
            model_spatial.fit(X_train_spatial, y_train_spatial)
            print("✅ Spatial model training complete!")
            
            y_pred_spatial = model_spatial.predict(X_test_spatial)
            y_pred_proba_spatial = model_spatial.predict_proba(X_test_spatial)[:, 1]
            
            accuracy_spatial = accuracy_score(y_test_spatial, y_pred_spatial)
            roc_auc_spatial = roc_auc_score(y_test_spatial, y_pred_proba_spatial)
            
            print("\n" + "="*60)
            print("SPATIAL BLOCK CROSS-VALIDATION RESULTS")
            print("="*60)
            print(f"Accuracy: {accuracy_spatial:.4f}")
            print(f"ROC-AUC Score: {roc_auc_spatial:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test_spatial, y_pred_spatial, 
                                       target_names=['Non-Floodplain', 'Floodplain']))
            
            cm_spatial = confusion_matrix(y_test_spatial, y_pred_spatial)
            print("\nConfusion Matrix:")
            print(f"True Negatives: {cm_spatial[0,0]:,} | False Positives: {cm_spatial[0,1]:,}")
            print(f"False Negatives: {cm_spatial[1,0]:,} | True Positives: {cm_spatial[1,1]:,}")
        else:
            print("⚠️ Empty training or testing set. Using random split as fallback.")
            accuracy_spatial = accuracy
            roc_auc_spatial = roc_auc
            y_pred_spatial = y_pred
            y_pred_proba_spatial = y_pred_proba
            cm_spatial = cm

# ---------- Compare with standard model (Task 2.1) ----------
print("\n" + "="*60)
print("COMPARISON: STANDARD vs SPATIAL BLOCK CV")
print("="*60)
print(f"{'Metric':<30} {'Standard (Random)':<20} {'Spatial Block CV':<20} {'Difference':<15}")
print("-"*85)
print(f"{'Accuracy':<30} {accuracy:<20.4f} {accuracy_spatial:<20.4f} {accuracy - accuracy_spatial:<15.4f}")
print(f"{'ROC-AUC':<30} {roc_auc:<20.4f} {roc_auc_spatial:<20.4f} {roc_auc - roc_auc_spatial:<15.4f}")

print("\n💡 INTERPRETATION:")
if accuracy > accuracy_spatial:
    print("✅ Standard accuracy > spatial CV accuracy → random split overestimates performance")
    print("   This confirms the need for spatial cross-validation in floodplain mapping")
else:
    print("⚠️ Spatial CV accuracy is similar to or higher than standard accuracy")
    print("   This suggests spatial autocorrelation may not be a major issue for this dataset")
print("="*60)

# ----------------------------------------------------------
# 9️⃣ Export predictions
# ----------------------------------------------------------

print("\n🗺️ Generating floodplain probability map...")

valid_mask_all = valid_features_mask
X_all_clean = X[valid_mask_all]

drainage_col_all = X_all_clean[:, drainage_index]
drainage_int_all = np.round(drainage_col_all).astype(int)
drainage_encoded_all = le.transform(drainage_int_all)

X_all_final = X_all_clean.copy()
X_all_final[:, drainage_index] = drainage_encoded_all

predicted_proba_clean = model.predict_proba(X_all_final)[:, 1]

predicted_proba_full = np.full(height * width, np.nan, dtype=np.float32)
predicted_proba_full[valid_mask_all] = predicted_proba_clean
predicted_proba = predicted_proba_full.reshape(height, width)

predicted_binary_clean = model.predict(X_all_final)
predicted_binary_full = np.full(height * width, np.nan, dtype=np.float32)
predicted_binary_full[valid_mask_all] = predicted_binary_clean
predicted_binary = predicted_binary_full.reshape(height, width)

out_meta = meta.copy()
out_meta.update({"dtype": "float32", "count": 1, "nodata": np.nan})

prob_path = RESULTS_DIR / "predicted_floodplain_probability.tif"
with rasterio.open(prob_path, "w", **out_meta) as dest:
    dest.write(predicted_proba.astype(np.float32), 1)

binary_path = RESULTS_DIR / "predicted_floodplain_binary.tif"
with rasterio.open(binary_path, "w", **out_meta) as dest:
    dest.write(predicted_binary.astype(np.float32), 1)

print(f"\n✅ Model finished! Files saved:")
print(f"   📊 Probability map: {prob_path}")
print(f"   🎯 Binary map: {binary_path}")

# ----------------------------------------------------------
# 🔟 Comparison plot
# ----------------------------------------------------------

binary_cmap = ListedColormap(['lightgray', 'red'])

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(target, cmap=binary_cmap, interpolation='none', vmin=0, vmax=1)
plt.title('Actual TRCA Floodplain')
legend_elements = [
    Patch(facecolor='red', label='Floodplain (1)'),
    Patch(facecolor='lightgray', label='Non-Floodplain (0)')
]
plt.legend(handles=legend_elements, loc='lower right')
plt.axis('off')

total_valid_pixels = np.sum(~np.isnan(target))
actual_floodplain_pct = np.sum(target == 1) / total_valid_pixels * 100

plt.text(0.02, 0.98, f"Actual:\n  Floodplain: {actual_floodplain_pct:.1f}%",
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
         verticalalignment='top', fontsize=9)

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

predicted_floodplain_pct = np.sum(predicted_binary == 1) / np.sum(~np.isnan(predicted_binary)) * 100
plt.text(0.02, 0.98, f"Predicted:\n  Floodplain: {predicted_floodplain_pct:.1f}%",
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
         verticalalignment='top', fontsize=9)

plt.suptitle(f'TRCA Floodplain: Actual vs Predicted\nAccuracy: {accuracy:.3f} | ROC-AUC: {roc_auc:.3f}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n📊 DETAILED STATISTICS:")
print("-" * 40)
print(f"ACTUAL FLOODPLAIN: {actual_floodplain_pct:.2f}% of area")
print(f"PREDICTED BINARY: {predicted_floodplain_pct:.2f}% of area")
print(f"Mean probability: {np.nanmean(predicted_proba):.4f}")

# Save the trained model
model_path = RESULTS_DIR / "random_forest_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Model saved to: {model_path}")
