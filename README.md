**Predictive Analytics group project** 
· Rebecca Quaye · Ketty Odero · Victoria Kutuusa

# What Makes Patients Recommend a Hospital?
Every US hospital that treats Medicare patients surveys its discharged patients with the national HCAHPS survey. One question
sums up the patient’s verdict: *would you recommend this hospital to your friends and family?* Across hospitals, the share who
answer “definitely yes” ranges from 30% to 94%.

This project finds which parts of a hospital stay are most linked to being recommended, predicts a hospital’s recommendation
rate, flags hospitals likely to fall below the national middle (70%), and groups hospitals into peer tiers. It ends with an
interactive Streamlit app.

**Main question:** which patient-experience dimensions are most important in predicting patients’ willingness to recommend a hospital?

## Key findings

- **Communication drives recommendation.** Nurse communication is the strongest driver of the recommendation rate; discharge
  information is the strongest sign that a hospital clears 70%; doctor communication is close behind.
- **Good hospitals are good at everything.** Hospitals fall into three quality tiers recommended at 78%, 70% and 61%.
- **Predictions are useful for planning.** Linear regression predicts a hospital’s rate within about 5 points on average
  (R² 0.56); the classifiers place about 3 in 4 hospitals on the right side of the 70% line.

## Data

CMS Provider Data Catalog, *Patient survey (HCAHPS) – Hospital* (dataset dgck-syfz), patients discharged October 2024 to
September 2025: https://data.cms.gov/provider-data/dataset/dgck-syfz. Public data, no NDA.
The cleaned file `data/hcahps_model_ready.csv` has one row per hospital: 3,172 hospitals with at least 100 completed surveys
and no CMS data-collection flags.

## Repository layout

```
data/            hcahps_model_ready.csv (final dataset)
notebooks/       HCAHPS_Group_Unified_Analysis.ipynb (EDA, PCA and clustering, regression, classification)
reports/         written report and presentation material
src/             [describe what is in this folder]
app.py           Streamlit app
train_models.py  trains the app’s models once and saves them to models/
models/          hcahps_models.joblib (saved, already-trained models)
.streamlit/      app theme (colours and sidebar)
```

## Methods

Six predictors (nurse communication, doctor communication, medicine explanation, discharge information, room cleanliness,
night quietness). Every supervised model learns from 60% of hospitals and is scored on the 40% it never saw
(`random_state=42`, stratified for classification), with settings tuned by 5-fold cross-validation on the training set only.

| Task | Models |
|---|---|
| Peer groups | PCA + k-means (3 tiers) |
| Predict the recommendation % | Multiple linear regression, decision tree |
| Flag hospitals below 70% | Logistic regression, classification tree, random forest |

## Running the analysis

```bash
pip install -r requirements.txt
jupyter notebook notebooks/HCAHPS_Group_Unified_Analysis.ipynb
```

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The trained models are already saved in `models/`, so the app does not retrain while you present. Run
`python train_models.py` only if the data changes.

**Pages:** Overview · Explore the data · What drives recommendation · Peer tiers · Hospital predictor (the MVP) ·
Test the models · How the models performed · Methods and limits.

## Limitations

Results show associations, not causes, from one year of hospital-level data. Hospitals with fewer than 100 surveys are
excluded, and factors such as hospital size, staffing and patient mix are not in the data.

## Use of AI

The group used AI assistants (ChatGPT and Claude) for coding, debugging, and merging notebooks.
Every number was produced by running the code on the real data and checked by the group.
