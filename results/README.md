```markdown
# Results Folder

This folder contains the output results from the floodplain mapping analysis.

## 📊 Performance Summary

| Metric | Don River | Humber River |
|--------|-----------|--------------|
| ROC-AUC | 0.986 | 0.802 |
| Accuracy | 93.3% | 87.5% |
| Floodplain Recall | 95.0% | 50.9% |
| Floodplain Precision | 50.0% | 26.1% |
| F1 Score | 0.61 | 0.345 |
| Optimal Threshold | 0.5 | 0.12 |

## 📁 Files

### Model
- `random_forest_model.pkl` - Trained Random Forest model

### Performance Metrics
- `model_performance_summary.csv` - All performance metrics
- `feature_importance.csv` - Feature importance scores
- `threshold_analysis.csv` - Threshold vs. F1 score data

### Don River Predictions
- `predicted_floodplain_probability.tif` - Don River probability map
- `predicted_floodplain_binary.tif` - Don River binary map (threshold 0.5)

### Humber River Predictions
- `predicted_floodplain_probability_humber.tif` - Humber probability map
- `predicted_floodplain_binary_humber.tif` - Humber binary map (threshold 0.12)

## 📈 Key Findings

- **Don River (Training):** ROC-AUC of 0.986, Accuracy of 93.3%
- **Humber River (Testing):** ROC-AUC of 0.802, Accuracy of 87.5%
- **Spatial Cross-Validation:** Random splits overestimated accuracy by 4.3%
- **Threshold Tuning:** Optimal threshold for Humber was 0.12 (F1 = 0.345)
- **Feature Importance:** Distance to river was the strongest predictor (50.3%)

## 📝 Notes

- Prediction rasters are in GeoTIFF format
- The model is saved as a pickle file for reproducibility
- See the main README for instructions on how to use these files
```