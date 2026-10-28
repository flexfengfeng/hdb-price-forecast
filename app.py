"""Streamlit app: forecast the median HDB resale price 6 or 12 months ahead.

Run:  streamlit run app.py
Needs best_model.pkl, created by the last section of hdb_price_forecast.ipynb.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "best_model.pkl"


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def forecast(bundle, h):
    """Forecast h months ahead with whichever kind of model was saved."""
    if bundle["kind"] == "drift":           # Naive (slope 0) or Drift
        return bundle["last_price"] + bundle["slope"] * np.arange(1, h + 1)
    if bundle["kind"] == "stats":           # fitted statsmodels result
        return np.asarray(bundle["model"].forecast(h))
    # ML model trained on monthly changes: predict one month at a time
    changes, price, preds = list(bundle["last_changes"]), bundle["last_price"], []
    for _ in range(h):
        next_change = bundle["model"].predict([changes[-bundle["n_lags"]:]])[0]
        changes.append(next_change)
        price += next_change
        preds.append(price)
    return np.array(preds)


st.set_page_config(page_title="HDB Price Forecast", page_icon="🏠")
st.title("🏠 HDB Resale Price Forecast")

if not MODEL_PATH.exists():
    st.error("best_model.pkl not found. Run hdb_price_forecast.ipynb first to train and save the model.")
    st.stop()

bundle = load_bundle()
history, results, method = bundle["history"], bundle["results"], bundle["method"]

st.write(f"Forecasting the **median resale price of all HDB flats** using **{method}**, "
         f"the method with the lowest backtest error. Data up to **{history.index[-1]:%b %Y}**.")

horizon = st.radio("Forecast horizon", [6, 12], format_func=lambda m: f"{m} months", horizontal=True)

preds = forecast(bundle, horizon)
future_index = pd.date_range(history.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
error_pct = results.loc[method, f"MAPE_{horizon}m"]

col1, col2, col3 = st.columns(3)
col1.metric(f"Latest ({history.index[-1]:%b %Y})", f"S${history.iloc[-1]:,.0f}")
col2.metric(f"Forecast ({future_index[-1]:%b %Y})", f"S${preds[-1]:,.0f}",
            delta=f"{(preds[-1] / history.iloc[-1] - 1) * 100:+.1f}%")
col3.metric("Typical backtest error", f"±{error_pct:.1f}%")

chart = pd.DataFrame({"History": history["2021":]})
chart = chart.reindex(chart.index.union(future_index))
chart.loc[future_index, "Forecast"] = preds
chart.loc[history.index[-1], "Forecast"] = history.iloc[-1]   # join the two lines
st.line_chart(chart, y_label="S$")

st.subheader("Monthly forecast")
table = pd.DataFrame({"Month": future_index.strftime("%b %Y"), "Forecast": preds,
                      "Low": preds * (1 - error_pct / 100), "High": preds * (1 + error_pct / 100)})
st.dataframe(table.style.format({c: "S${:,.0f}" for c in ["Forecast", "Low", "High"]}),
             hide_index=True, width="stretch")
st.caption("Low / High = forecast ± the method's average backtest error. A rough guide, not a guarantee.")

with st.expander("How the methods compared (backtest)"):
    st.dataframe(results.style.format({"MAE_6m": "S${:,.0f}", "MAE_12m": "S${:,.0f}",
                                       "MAPE_6m": "{:.2f}%", "MAPE_12m": "{:.2f}%"}),
                 width="stretch")
    st.caption("Average error over three forecast origins. Lower is better. Ranked by 12-month MAPE.")
