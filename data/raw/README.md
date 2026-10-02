# Raw data

The raw file is **not stored in this repo** because it is about 105 MB, above GitHub's 100 MB file limit.

## How to get it

1. Go to the [CMS Provider Data Catalog: Patient survey (HCAHPS) – Hospital](https://data.cms.gov/provider-data/dataset/dgck-syfz).
2. Download the CSV (this project uses the release covering discharges Oct 2024 – Sep 2025).
3. Save it in this folder as `HCAHPS-Hospital.csv`.

## What it looks like

- 325,720 rows and 22 columns, covering 4,790 hospitals.
- **Long format:** about 68 rows per hospital, one for each survey measure and answer choice.
- Each row stores its value in one of three columns, depending on the measure type:

| Measure ID ending | Meaning | Value column |
|---|---|---|
| `_A_P`, `_U_P`, `_SN_P`, `_Y_P`, `_N_P` | % answering Always / Usually / Sometimes-or-Never / Yes / No | `HCAHPS Answer Percent` |
| `_LINEAR_SCORE` | 0–100 average score | `HCAHPS Linear Mean Value` |
| `_STAR_RATING` | 1–5 stars | `Patient Survey Star Rating` |

"Not Applicable" means the column doesn't apply to that row type. It doesn't mean the value is missing.

The cleaned, one-row-per-hospital version is in `data/processed/hcahps_model_ready.csv`.
