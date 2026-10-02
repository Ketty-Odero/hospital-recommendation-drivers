# What Makes Patients Recommend a Hospital?

**Predicting patient experience across 3,183 U.S. hospitals with regression, classification and clustering on 2024–25 CMS survey data.**

>**Status:** in progress. Data cleaning is complete; modeling and the Streamlit app are under way.

---

## Overview

Hospitals have limited budgets and can't fix everything at once. Patient-experience scores affect their Medicare payments (Medicare's Hospital Value-Based Purchasing program weights patient experience at 25%) and their public star ratings. Hospitals need to know **which improvements will raise patient recommendations the most**.

This project uses the CMS HCAHPS patient survey to find which parts of the hospital stay best predict whether patients would "definitely recommend" a hospital. It then turns those findings into priorities for each type of hospital.

## Research questions

**Descriptive**
- **D1.** Which parts of the stay (e.g. nurse communication, quiet at night) get the lowest scores nationally?
- **D2.** How do hospitals that most patients would recommend differ from the rest?
- **D3.** Do scores differ by U.S. region and hospital size?

**Predictive**
- **P1.** Which parts of the stay best predict the % of patients who would definitely recommend a hospital?
- **P2.** Can experience scores classify a hospital as high-recommendation (70%+)?
- **P3.** Do hospitals fall into types with shared strengths and weaknesses, and what should each type fix first?

## Data

| | |
|---|---|
| **Source** | [CMS Provider Data Catalog: Patient survey (HCAHPS) – Hospital](https://data.cms.gov/provider-data/dataset/dgck-syfz) |
| **Period** | Patients discharged Oct 2024 – Sep 2025 |
| **Raw** | 325,720 rows (one row per hospital × survey answer), 4,790 hospitals |
| **Cleaned** | 3,183 hospitals × 31 columns (one row per hospital, at least 100 completed surveys) |

The raw file (~105 MB) is larger than GitHub's 100 MB file limit, so it isn't stored in this repo. See [`data/raw/README.md`](data/raw/README.md) for how to download it.

### Key variables

| Variable | Description | Role |
|---|---|---|
| `recommend_pct` | % of patients who would "definitely" recommend the hospital | Continuous target |
| `high_recommend` | 1 if `recommend_pct` ≥ 70 (the median), else 0 | Categorical target |
| `nurse_comm`, `doctor_comm`, `medicine_comm`, `cleanliness`, `quietness`, `discharge_info` | % of patients answering "always" ("yes" for discharge info) | Predictors |
| `region`, `large_sample`, `response_rate` | Census region; 1 if ≥ 300 surveys; survey response rate | Controls |

## Methods

| Task | Methods | Evaluation |
|---|---|---|
| Unsupervised | PCA; k-means clustering on each hospital's relative strengths | Variance explained; elbow method and silhouette score |
| Classification (`high_recommend`) | Logistic regression; classification tree | Accuracy, recall, confusion matrix, ROC AUC |
| Regression (`recommend_pct`) | Multiple linear regression; regression tree | RMSE, adjusted R², VIF and residual checks |

All supervised models use a 70/30 train-test split with 5-fold cross-validation. Columns that restate the target (`recommend_linear`, `overall_rating_9_10`, `summary_star_rating`) are excluded to prevent data leakage.

## Results

_Coming soon._

## Interactive app

_Coming soon:_ a Streamlit app where hospital managers can enter their scores, see a predicted recommend rate, test "what-if" improvements and benchmark against similar hospitals.

## Project structure

```
hospital-recommendation-drivers/
├── data/
│   ├── raw/              # raw CMS file (not committed; see data/raw/README.md)
│   └── processed/        # cleaned, model-ready dataset
├── notebooks/            # analysis notebooks, run in numbered order
├── src/                  # reusable Python code (cleaning, features, models)
├── models/               # saved trained models (.joblib)
├── app/                  # Streamlit app
├── reports/
│   ├── figures/          # charts used in the README and report
│   └── project_proposal.docx
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/<your-username>/hospital-recommendation-drivers.git
cd hospital-recommendation-drivers
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                      # open the notebooks in notebooks/
```

## Limitations

- **Association, not causation.** The predictors and the target come from the same patient survey.
- **Small hospitals are excluded.** Hospitals with fewer than 100 surveys were dropped for reliability.
- **Hospital-level data.** Results describe hospitals, not individual patients.

## Author

**[Ketty Odero]**: [LinkedIn](https://www.linkedin.com/in/kettyaldy/) · [Email](mailto:you@example.com)
