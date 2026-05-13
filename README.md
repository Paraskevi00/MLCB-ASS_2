# MLCB 2026 Assignment 2 — Heart Disease Classification

Leakage-safe implementation of repeated nested cross-validation for the UCI Cleveland Heart Disease dataset.

## Structure

```text
src/
  preprocessing.py      # raw-data loading, ColumnTransformer, pipelines, model registry
  rncv.py               # OOP repeated nested CV implementation
  final_model.py        # final CV search, complete-pipeline saving, SHAP, error analysis
  plotting.py           # reusable figure helpers
notebooks/
  01_EDA.ipynb
  02_rnCV_comparison.ipynb
  03_final_model_shap.ipynb
data/
  heart_disease.csv
models/
figures/
results/
```

## Leakage controls

- No imputation, scaling, encoding, or feature selection is fitted before CV.
- Raw `X` is passed into sklearn `Pipeline` objects.
- `RandomizedSearchCV` receives the complete pipeline, so preprocessing is refit inside every inner fold.
- Feature selection, when used, is inside the pipeline and is refit within CV folds.
- The saved model is a complete raw-input pipeline: preprocessing + optional feature selection + classifier.

## Run order

1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_rnCV_comparison.ipynb`
3. `notebooks/03_final_model_shap.ipynb`

For quick testing, reduce `N_ROUNDS`, `N_ITER`, or the estimator list in notebook 02. For final submission, use `R=10`, `N=5`, `K=3`.
