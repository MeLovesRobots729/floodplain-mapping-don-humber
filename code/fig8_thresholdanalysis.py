# AI assistance: this file was written with the assistance of DeepSeek V3 (2026).
 # The code was reviewed, tested, and modified by Alexander Zhang.

import matplotlib.pyplot as plt
import numpy as np
from paths import FIGURES_DIR

# Your threshold analysis data
thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
f1_scores = [0.122, 0.125, 0.174, 0.228, 0.284, 0.326, 0.341, 0.345, 0.336, 0.304, 0.264, 0.220]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the line with markers
ax.plot(thresholds, f1_scores, 'o-', color='blue', linewidth=2, markersize=8, label='F1 Score')

# Mark the optimal threshold
optimal_threshold = 0.12
optimal_f1 = 0.345
ax.plot(optimal_threshold, optimal_f1, 'o', color='red', markersize=12, 
        label=f'Optimal: {optimal_threshold} (F1 = {optimal_f1:.3f})')

# Add vertical dashed line at optimal threshold
ax.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7)

# Labels and title
ax.set_xlabel('Probability Threshold', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 Score vs. Threshold (Humber River)', fontsize=14, fontweight='bold')

# Grid
ax.grid(True, alpha=0.3)

# Legend
ax.legend(loc='best', fontsize=10)

# Set x-axis limits
ax.set_xlim(0, 0.32)
ax.set_ylim(0, 0.40)

# Add a text annotation for the optimal threshold
ax.annotate(f'Optimal Threshold = 0.12\nF1 Score = 0.345',
            xy=(optimal_threshold, optimal_f1),
            xytext=(0.18, 0.32),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Tight layout and save
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure8_threshold_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Figure 8 saved as: {FIGURES_DIR / 'figure8_threshold_analysis.png'}")
plt.close()
