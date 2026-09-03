# Results

This folder contains the output results from the floodplain mapping analysis.

## Performance Metrics

| Metric | Don River | Humber River |
|--------|-----------|--------------|
| ROC-AUC | 0.986 | 0.802 |
| Accuracy | 93.3% | 87.5% |
| Floodplain Recall | 95.0% | 50.9% |
| Floodplain Precision | 50.0% | 26.1% |
| F1 Score | 0.61 | 0.345 |
| Optimal Threshold | 0.5 | 0.12 |

## Files

- `model_performance_summary.csv` - All performance metrics
- `feature_importance.csv` - Feature importance scores
- `threshold_analysis.csv` - Threshold vs. F1 score data
- `don_confusion_matrix.csv` - Confusion matrix for Don River
- `humber_confusion_matrix.csv` - Confusion matrix for Humber River

## Notes

- Full prediction rasters are not included due to file size limits
- The trained Random Forest model (`random_forest_model.pkl`) is included for reproducibility
