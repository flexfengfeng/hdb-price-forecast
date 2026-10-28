# HDB Resale Price Forecast (6 and 12 months)

Forecast the **median HDB resale price** for the next 6 and 12 months. Compare statistical
time-series models with machine-learning models, then deploy the best one with Streamlit.

## What's inside

| File | What it does |
|---|---|
| `download_data.py` | Downloads the latest HDB resale data from [data.gov.sg](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view) into `data/hdb_resale.csv` |
| `hdb_price_forecast.ipynb` | **The lesson.** Builds the monthly series, compares 8 methods, saves the winner |
| `best_model.pkl` | The saved winning model (created by the notebook) |
| `app.py` | Streamlit app that loads `best_model.pkl` and shows the forecast |

## Quick start

```bash
pip install -r requirements.txt
python download_data.py          # optional: refresh data (a snapshot is already in data/)
jupyter notebook hdb_price_forecast.ipynb    # run all cells, this saves best_model.pkl
streamlit run app.py
```

## Methods compared

**Statistical time-series:** Naive (baseline), Drift, Holt (ETS), Holt-Winters, SARIMA

**Machine learning (lag features, recursive forecasting):** Linear Regression, Random Forest, Gradient Boosting

Every method is **backtested from 3 forecast origins**. We measure MAE and MAPE over the first 6 and
all 12 forecast months. The method with the lowest **12-month MAPE** is deployed.

## Results (data up to Sep 2026)

| Method | MAPE 6m | MAPE 12m |
|---|---|---|
| **Naive** | 1.83% | **2.00%** |
| Drift | 1.88% | 2.54% |
| Random Forest | 1.93% | 2.77% |
| Holt-Winters | **1.76%** | 2.97% |
| Holt (ETS) | 1.94% | 3.09% |
| Linear Regression | 2.07% | 3.11% |
| Gradient Boosting | 2.55% | 3.37% |
| SARIMA | 2.60% | 3.87% |

Prices rose fast from 2020 to 2024 and flattened in 2025-2026. Models that learned the old uptrend
over-forecast, so the simple Naive baseline wins at 12 months. The ranking can change when you
re-run with newer data. The notebook always deploys whichever method wins.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (include `best_model.pkl`).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, and pick the repo with `app.py` as the main file.

To update the model: run `download_data.py`, re-run the notebook, then commit the new `best_model.pkl`.
