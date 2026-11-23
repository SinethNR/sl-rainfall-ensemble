"""Wesi: Sri Lanka Rainfall Ensemble Lab (Streamlit dashboard).

Scaffold only. The full dashboard (overview, model arena, explainability,
live predictor) is built in Stage 5 once the notebook has written models/.

GenAI declaration: this file was written with the help of Claude (Anthropic).
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sl_weather_2015_2025.csv"

st.set_page_config(page_title="Wesi: Sri Lanka Rainfall Ensemble Lab", page_icon="🌧️", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


st.title("🌧️ Wesi: Sri Lanka Rainfall Ensemble Lab")
st.caption("Predicting tomorrow's rainfall in Colombo, Kandy and Galle with Bagging, Boosting, Stacking and RF + ARIMA.")

df = load_data()
c1, c2, c3 = st.columns(3)
c1.metric("Days of real ERA5 data", f"{df['date'].nunique():,}")
c2.metric("Cities", df["city"].nunique())
c3.metric("Period", f"{df['date'].min():%Y} to {df['date'].max():%Y}")

st.info("Dashboard under construction. Models will appear after the notebook is run.")

with st.expander("GenAI Usage Declaration"):
    st.write(
        "Code and UI for this dashboard were written with the help of Claude (Anthropic) and reviewed by the author. "
        "The dataset is real ERA5 reanalysis weather from Open-Meteo, not AI-generated."
    )
