# Notebooks

Run these in order. Each one reads from `data/processed/` and saves its outputs to `models/` or `reports/figures/`.

| Notebook | Purpose | Answers |
|---|---|---|
| `01_data_cleaning.ipynb` | Reshape the raw long file into one row per hospital; build the derived columns | — |
| `02_eda.ipynb` | Distributions, group comparisons, correlations, regional differences | D1–D3 |
| `03_unsupervised.ipynb` | PCA and k-means clustering on relative strengths | P3 |
| `04_classification.ipynb` | Logistic regression and classification tree for `high_recommend` | P2 |
| `05_regression.ipynb` | Linear regression and regression tree for `recommend_pct` | P1 |
| `06_model_export.ipynb` | Save final models with `joblib` for the Streamlit app | — |
