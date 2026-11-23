"""වැසි: Sri Lanka Rainfall Ensemble Lab (Streamlit dashboard).

Loads the models and results saved by ensemble_analysis.ipynb (models/), so it starts
instantly and never retrains.

GenAI declaration: this dashboard's code and UI were written with the help of
Claude (Anthropic) and reviewed by the author. The data is real ERA5 reanalysis
weather from Open-Meteo, not AI-generated.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPO_URL = "https://github.com/SinethNR/sl-rainfall-ensemble"
LIVE_URL = "https://wesi-sl-rainfall.streamlit.app"
HEAVY = 10.0

st.set_page_config(page_title="වැසි: Sri Lanka Rainfall Ensemble Lab", page_icon="🌧️", layout="wide")

CITY_COLORS = {"Colombo": "#1f77b4", "Kandy": "#2ca02c", "Galle": "#ff7f0e"}
FAMILY_COLORS = {"Baseline": "#9e9e9e", "Neural net": "#8c564b", "Bagging": "#1f77b4", "Boosting": "#2ca02c",
                 "Stacking": "#d62728", "Time-series": "#9467bd", "Hybrid": "#ff7f0e"}
SEASONS = ["NE monsoon (Maha)", "First inter-monsoon", "SW monsoon (Yala)", "Second inter-monsoon"]
ML_MODELS = ["Ridge (linear)", "Random Forest (Bagging)", "LightGBM (Boosting)", "XGBoost (Boosting)",
             "CatBoost (Boosting)", "ANN (MLP)", "Stacking (RF+LGBM+Cat+ANN)"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Sinhala:wght@600;700&display=swap');
.block-container {padding-top: 1.6rem;}
.si {font-family: 'Noto Sans Sinhala', 'Nirmala UI', 'Iskoola Pota', sans-serif;}
[data-testid="stTab"], .stTabs [data-baseweb="tab"] {margin-right: 1.1rem;}
[data-testid="stTab"] p, .stTabs [data-baseweb="tab"] p {font-size: 1.1rem !important; font-weight: 600 !important;}
@media (max-width: 640px) {
  .block-container {padding-left: .8rem; padding-right: .8rem; padding-top: 1rem;}
  .hero {padding: 1rem 1.1rem;}
  .hero h1 {font-size: 1.45rem !important; line-height: 1.25 !important;}
  .kpi {margin-bottom: .5rem;}
  [data-testid="stTab"], .stTabs [data-baseweb="tab"] {margin-right: .35rem;}
  [data-testid="stTab"] p, .stTabs [data-baseweb="tab"] p {font-size: .95rem !important;}
}
.hero {background: linear-gradient(120deg, #0b3d5c 0%, #136f63 55%, #e0a100 140%); color: #fff;
       padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;}
.hero h1 {color: #fff; margin: 0 0 .25rem 0; font-size: 2.0rem;}
.hero p {margin: 0; opacity: .92;}
.kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .8rem 1rem; height: 100%;}
.kpi .v {font-size: 1.6rem; font-weight: 700;}
.kpi .l {font-size: .82rem; opacity: .75;}
.callout {border-left: 5px solid #d62728; background: rgba(214,39,40,.07); padding: .8rem 1rem; border-radius: 8px;}
.note {border-left: 5px solid #136f63; background: rgba(19,111,99,.08); padding: .8rem 1rem; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
@st.cache_data
def load_tables():
    raw = pd.read_csv(DATA / "sl_weather_2015_2025.csv", parse_dates=["date"])
    feat = pd.read_csv(DATA / "sl_weather_features.csv", parse_dates=["date"])
    preds = pd.read_csv(MODELS / "test_predictions.csv", parse_dates=["date"])
    metrics = json.loads((MODELS / "metrics.json").read_text())
    shap_imp = pd.read_csv(MODELS / "shap_importance.csv", index_col=0)["mean_abs_shap"]
    shap_sample = pd.read_csv(MODELS / "shap_sample.csv")
    rf_imp = pd.read_csv(MODELS / "rf_importance.csv", index_col=0)["rf_importance"]
    return raw, feat, preds, metrics, shap_imp, shap_sample, rf_imp


@st.cache_resource
def load_models(files):
    models = {name: joblib.load(MODELS / f) for name, f in files.items()}
    arima = {c: joblib.load(MODELS / f"arima_{c.lower()}.joblib").params for c in CITY_COLORS}
    return models, arima


raw, feat, preds, M, shap_imp, shap_sample, rf_imp = load_tables()
models, arima_params = load_models(M["model_files"])
NUM, CAT = M["features"]["numeric"], M["features"]["categorical"]
FEATURES = NUM + CAT
SMEAR = M["smearing_factors"]
test_tbl = pd.DataFrame(M["test"]).T
for c in test_tbl.columns.drop("Family"):
    test_tbl[c] = test_tbl[c].astype(float)
test_tbl = test_tbl.sort_values("RMSE_mm")
BEST = M["best_model"]

# ---------------------------------------------------------------- header
st.markdown(f"""
<div class="hero">
<h1>🌧️ <span class="si">වැසි</span>: Sri Lanka Rainfall Ensemble Lab</h1>
<p>How much rain will fall <b>tomorrow</b> in Colombo, Kandy and Galle? Bagging, Boosting, Stacking and an RF + ARIMA hybrid,
trained on 11 years of <b>real ERA5 reanalysis</b> weather and tested on 2024–2025.</p>
</div>""", unsafe_allow_html=True)

kpis = [(f"{raw['date'].nunique():,}", "days of real weather × 3 cities"),
        ("11", "models compared (bagging, boosting, stacking, hybrid, baselines)"),
        (f"{test_tbl.loc[BEST, 'R2']:.3f}", f"best test R² ({BEST.split(' (')[0]})"),
        (f"{test_tbl.loc[BEST, 'RMSE_gain_vs_persistence_%']:.0f}%", "lower RMSE than 'tomorrow = today'"),
        (f"{test_tbl.loc[BEST, 'HeavyRain_AUC']:.2f}", "heavy-rain (≥10 mm) AUC")]
for col, (v, l) in zip(st.columns(5), kpis):
    col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["🌦️ Overview & climate", "🏆 Model arena", "🔍 Why? Explainability", "🔮 Live predictor"])

# ================================================================ TAB 1
with tab1:
    left, right = st.columns([1.55, 1])
    with left:
        st.subheader("A two-peak rainfall year")
        clim = raw.assign(month=raw["date"].dt.month).groupby(["city", "month"])["precipitation_sum"].mean().reset_index()
        fig = px.line(clim, x="month", y="precipitation_sum", color="city", markers=True, color_discrete_map=CITY_COLORS,
                      labels={"precipitation_sum": "mean daily rain (mm)", "month": ""})
        for name, a, b in [("NE monsoon", 0.5, 2.5), ("1st inter-monsoon", 2.5, 4.5), ("SW monsoon (Yala)", 4.5, 9.5),
                           ("2nd inter-monsoon", 9.5, 11.5)]:
            fig.add_vrect(x0=a, x1=b, fillcolor="grey", opacity=0.07 if "inter" in name else 0.13, line_width=0,
                          annotation_text=name, annotation_position="top left", annotation_font_size=10)
        fig.update_layout(height=380, xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=list("JFMAMJJASOND")),
                          legend=dict(orientation="h", y=-0.15), margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Colombo and Galle peak in May (SW monsoon onset) and Oct–Nov (second inter-monsoon). Kandy sits in the "
                   "central hills, partly sheltered from the SW monsoon, and gets most of its rain in Oct–Nov.")
    with right:
        st.subheader("Where the data comes from")
        loc = raw.groupby("city").agg(lat=("latitude", "first"), lon=("longitude", "first"), elev=("elevation_m", "first"),
                                      rain=("precipitation_sum", "mean")).reset_index()
        try:
            fig = px.scatter_map(loc, lat="lat", lon="lon", size="rain", color="city", color_discrete_map=CITY_COLORS,
                                 hover_data={"elev": True, "rain": ":.1f"}, zoom=6.4, center={"lat": 7.4, "lon": 80.6},
                                 map_style="open-street-map", size_max=28)
        except Exception:
            fig = px.scatter_geo(loc, lat="lat", lon="lon", size="rain", color="city", color_discrete_map=CITY_COLORS)
        fig.update_layout(height=380, margin=dict(t=10, b=10, l=0, r=0), showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Bubble size = mean daily rain. Kandy ≈ 500 m elevation; Colombo and Galle are on the coast.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Chance of a heavy-rain tomorrow (≥ 10 mm)")
        hm = feat.pivot_table(index="city", columns="monsoon_season", values="rain_tomorrow_event", aggfunc="mean")[SEASONS] * 100
        fig = px.imshow(hm.round(0), text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                        labels={"color": "% of days", "x": "", "y": ""})
        fig.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Season × city interaction: Colombo jumps from 10% (Maha) to 44% (Oct–Nov). Tree ensembles learn this.")
    with c2:
        st.subheader("Annual rainfall totals")
        annual = raw.assign(year=raw["date"].dt.year).groupby(["year", "city"])["precipitation_sum"].sum().reset_index()
        fig = px.bar(annual, x="year", y="precipitation_sum", color="city", barmode="group", color_discrete_map=CITY_COLORS,
                     labels={"precipitation_sum": "mm", "year": ""})
        fig.update_layout(height=300, margin=dict(t=10, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
        st.caption("2016 was the driest year (the 2016–17 drought), yet it also holds the wettest day in the record.")

    st.markdown("""
<div class="callout"><b>🌀 Real-data proof: the record captures Sri Lanka's cyclones</b><br>
<b>15 May 2016:</b> 231 mm in Colombo and 173 mm in Kandy, the depression that became <b>Cyclone Roanu</b> (Aranayake landslide, Colombo floods).<br>
<b>27–28 Nov 2025:</b> 217 mm in Colombo and 165 mm in Kandy, during <b>Cyclone Ditwah</b>. These days fall inside our 2024–2025 test set,
the hardest possible exam for the models.</div>""", unsafe_allow_html=True)
    st.write("")
    top = raw.nlargest(10, "precipitation_sum")[["date", "city", "precipitation_sum", "precipitation_hours", "pressure_msl_mean"]]
    top.columns = ["Date", "City", "Rain (mm)", "Rain hours", "Pressure (hPa)"]
    top["Date"] = top["Date"].dt.date
    with st.expander("The 10 wettest days in the dataset"):
        st.dataframe(top, hide_index=True, width="stretch")
    with st.expander("About the data (ERA5 reanalysis, real and reproducible)"):
        st.markdown(f"""
- **Source:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), backed by **ECMWF ERA5 / ERA5-Land**.
- **What reanalysis is:** millions of real observations (stations, balloons, aircraft, ships, satellites) blended with a physics model into a
  gap-free ~9–31 km grid. Observation-based, **not synthetic**. Limitation: it smooths extreme cloudbursts and over-counts drizzle.
- **Coverage:** 2015-01-01 to 2025-12-31, daily, 0 missing values. Re-download any time with `python scripts/fetch_data.py`.
- **Split:** train 2015–2022 · validation 2023 · **test 2024–2025** (time-based, no shuffling, no leakage).
- Code: [{REPO_URL}]({REPO_URL})
""")

# ================================================================ TAB 2
with tab2:
    st.subheader("Test-set leaderboard (2024–2025, 2,190 unseen city-days, metrics in mm)")
    fams = st.multiselect("Families", list(FAMILY_COLORS), default=list(FAMILY_COLORS))
    show = test_tbl[test_tbl["Family"].isin(fams)].copy()
    show_disp = show[["Family", "RMSE_mm", "MAE_mm", "R2", "HeavyRain_AUC", "HeavyRain_F1", "RMSE_gain_vs_persistence_%"]]
    show_disp.columns = ["Family", "RMSE (mm) ↓", "MAE (mm) ↓", "R² ↑", "Heavy-rain AUC ↑", "Heavy-rain F1 ↑", "RMSE gain vs persistence %"]
    st.dataframe(show_disp.style.format(precision=3).background_gradient(subset=["R² ↑"], cmap="Greens")
                 .background_gradient(subset=["RMSE (mm) ↓"], cmap="Greens_r"), width="stretch", height=422)

    metric = st.radio("Metric", ["RMSE_mm", "MAE_mm", "R2", "HeavyRain_AUC"], horizontal=True,
                      format_func=lambda m: {"RMSE_mm": "RMSE (mm)", "MAE_mm": "MAE (mm)", "R2": "R²", "HeavyRain_AUC": "Heavy-rain AUC"}[m])
    asc = metric in ("RMSE_mm", "MAE_mm")
    s = show.sort_values(metric, ascending=not asc).reset_index(names="Model")
    METRIC_LABELS = {"RMSE_mm": "RMSE (mm), lower is better", "MAE_mm": "MAE (mm), lower is better",
                     "R2": "R², higher is better", "HeavyRain_AUC": "Heavy-rain AUC, higher is better"}
    fig = px.bar(s, x=metric, y="Model", color="Family", orientation="h", color_discrete_map=FAMILY_COLORS, text=s[metric].round(3),
                 labels={metric: METRIC_LABELS[metric], "Model": ""})
    fig.update_layout(height=470, yaxis=dict(categoryorder="array", categoryarray=s["Model"].tolist()), margin=dict(t=60),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title_text=""))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Is the winner really better? Paired bootstrap vs {BEST.split(' (')[0]}**")
        bs = pd.DataFrame(M["bootstrap_vs_best"]).T
        bs = bs[~bs.index.isin(["Persistence", "Climatology"])].astype({"mean_RMSE_gap_mm": float, "CI95_low": float, "CI95_high": float})
        bs = bs.sort_values("mean_RMSE_gap_mm")
        fig = go.Figure(go.Scatter(x=bs["mean_RMSE_gap_mm"], y=bs.index, mode="markers",
                                   marker=dict(size=11, color=np.where(bs["CI95_low"] > 0, "#d62728", "#2ca02c")),
                                   error_x=dict(type="data", symmetric=False, array=bs["CI95_high"] - bs["mean_RMSE_gap_mm"],
                                                arrayminus=bs["mean_RMSE_gap_mm"] - bs["CI95_low"])))
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=360, xaxis_title=f"RMSE gap vs {BEST.split(' (')[0]} (mm), 95% CI", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        st.markdown('<div class="note">🟢 interval crosses 0 = <b>statistically tied</b> with the winner. 🔴 = significantly worse. '
                    'Random Forest, Stacking and CatBoost are tied with XGBoost: the top ensembles are equally good within noise.</div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown("**The log-bias fix (Duan smearing, fitted on 2023 only)**")
        vs = pd.DataFrame(M["validation_smearing"]).T.sort_values("R2_after")
        fig = go.Figure([go.Bar(y=vs.index, x=vs["R2_before"], name="raw expm1", orientation="h", marker_color="#bbbbbb"),
                         go.Bar(y=vs.index, x=vs["R2_after"], name="after smearing", orientation="h", marker_color="#136f63")])
        fig.update_layout(barmode="group", height=360, xaxis_title="validation R²", margin=dict(t=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
        st.caption("Models trained on log(rain) predict a 'typical' day, about 1.5× too low in mm. One frozen factor per model fixes it.")

    st.markdown("**Forecast vs reality on the test period**")
    cc1, cc2, cc3 = st.columns([1, 1, 2])
    city_sel = cc1.selectbox("City", list(CITY_COLORS), key="ts_city")
    period = cc2.selectbox("Period", ["Oct–Dec 2025 (Cyclone Ditwah)", "May–Jun 2024 (SW monsoon onset)", "All 2024–2025"])
    models_sel = cc3.multiselect("Models", [BEST, "Stacking (RF+LGBM+Cat+ANN)", "Hybrid RF + ARIMA", "Persistence"],
                                 default=[BEST, "Stacking (RF+LGBM+Cat+ANN)"])
    p = preds[preds.city == city_sel].copy()
    p["target_date"] = p["date"] + pd.Timedelta(days=1)
    rng = {"Oct–Dec 2025 (Cyclone Ditwah)": ("2025-10-01", "2025-12-31"), "May–Jun 2024 (SW monsoon onset)": ("2024-05-01", "2024-06-30"),
           "All 2024–2025": ("2024-01-01", "2025-12-31")}[period]
    p = p[(p.target_date >= rng[0]) & (p.target_date <= rng[1])]
    fig = go.Figure(go.Bar(x=p["target_date"], y=p["rain_tomorrow_mm"], name="actual", marker_color="rgba(31,119,180,.45)"))
    for m_ in models_sel:
        fig.add_trace(go.Scatter(x=p["target_date"], y=p[m_], name=m_, mode="lines"))
    fig.update_layout(height=360, yaxis_title="mm", margin=dict(t=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Error by city and monsoon season (RMSE, mm)**")
    seg = pd.DataFrame(M["segment_rmse"]).T
    seg.index = [i.split("|")[1] for i in seg.index]
    fig = px.imshow(seg.round(2), text_auto=".2f", color_continuous_scale="Reds", aspect="auto")
    fig.update_layout(height=330, margin=dict(t=10), coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("The second inter-monsoon (Oct–Nov) is hardest for every model: it holds the convective storms and Cyclone Ditwah.")

# ================================================================ TAB 3
with tab3:
    st.subheader("What drives tomorrow's rain? (SHAP on LightGBM, test set)")
    c1, c2 = st.columns([1, 1.25])
    with c1:
        top15 = shap_imp.head(15)[::-1]
        fig = px.bar(x=top15.values, y=top15.index, orientation="h", labels={"x": "mean |SHAP| (log-rain)", "y": ""},
                     color=top15.values, color_continuous_scale="Teal")
        fig.update_layout(height=470, coloraxis_showscale=False, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        feats = [c[3:] for c in shap_sample.columns if c.startswith("s__")][:10]
        rows = []
        for f in feats:
            x = shap_sample[f"x__{f}"]
            rows.append(pd.DataFrame({"feature": f, "shap": shap_sample[f"s__{f}"],
                                      "value (scaled)": (x.rank(pct=True)).values}))
        bee = pd.concat(rows)
        fig = px.scatter(bee, x="shap", y="feature", color="value (scaled)", color_continuous_scale="RdBu_r",
                         category_orders={"feature": feats}, opacity=0.55)
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=470, margin=dict(t=10), xaxis_title="SHAP value (impact on log-rain forecast)", yaxis_title="",
                          coloraxis_colorbar=dict(title="feature<br>value", tickvals=[0, 1], ticktext=["low", "high"]))
        st.plotly_chart(fig, width="stretch")
    st.caption("Red = high feature value, blue = low. Lots of rain today, many rain hours and rain in Colombo/Kandy push tomorrow's "
               "forecast up; high pressure and strong sunshine push it down.")

    st.subheader("Why the models behaved the way they did")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("""
**🏆 Why XGBoost led (boosting = lower bias)**
- Shallow trees added one after another, each fixing the last one's errors, so it lowers **bias**.
- It reaches further into the tail: its biggest forecast was **66 mm**, vs 52 mm for Random Forest and 50 mm for Stacking.
- Rain is **non-linear**: today's rain saturates, humidity acts like a switch above ~88%, and falling pressure warns of depressions. Trees capture all three natively.
- But Random Forest (**best MAE**, 5.42 mm) and Stacking are **statistically tied** with it (bootstrap).
""")
    with w2:
        st.markdown("""
**🧱 Why Stacking flattened the cyclone peaks**
- It **won validation (2023)** and has the **best heavy-rain AUC** on test (0.825).
- But its base models (RF, LightGBM, CatBoost) make **very similar errors**, so there is little diversity to exploit.
- **Averaging conservative models gives an even more conservative forecast**: about 48 mm predicted for Ditwah's 217 mm day.
- Only **22 extreme days** cause **~45% of all squared error**, so flattening them costs RMSE.
""")
    with w3:
        st.markdown("""
**🔗 How RF + ARIMA beat ARIMA alone**
- **ARIMA** (per city, AR + annual Fourier terms) captures the *linear* part: persistence and monsoon seasonality.
- **Random Forest on ARIMA's residuals** learns the *non-linear* part: humidity, wind, pressure and cross-city rain.
- RMSE **10.27 → 10.02 mm**, R² **0.276 → 0.309**: the Zhang hybrid works.
- It doesn't beat direct ensembles, because they already get lag and rolling features (they already "contain" ARIMA).
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stacking meta-learner weights (how much each base model is trusted)**")
        sw = pd.Series(M["stacking_weights"])
        fig = px.pie(values=sw.values, names=sw.index, hole=0.5, color=sw.index,
                     color_discrete_map={"Random Forest": "#1f77b4", "LightGBM": "#2ca02c", "CatBoost": "#17becf", "ANN": "#8c564b"})
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("CatBoost carries the stack; the ANN earns 23% because its errors differ from the trees' errors (diversity).")
    with c2:
        st.markdown("**A few cyclone days dominate RMSE**")
        ex = pd.DataFrame(M["extreme_days"]).T.astype(float)
        ex = ex[ex.index.isin(ML_MODELS + ["Hybrid RF + ARIMA", "ARIMA (per city)"])].sort_values("RMSE_all_days")
        fig = go.Figure([go.Bar(x=ex.index, y=ex["RMSE_all_days"], name="all days", marker_color="#d62728"),
                         go.Bar(x=ex.index, y=ex["RMSE_without_top1pct"], name="without 22 extreme days", marker_color="#136f63")])
        fig.update_layout(barmode="group", height=340, yaxis_title="RMSE (mm)", margin=dict(t=50), legend=dict(orientation="h", y=1.04, yanchor="bottom", x=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("On ordinary days the ensembles' RMSE is only ~7.3 mm. Trees cannot predict beyond what they have seen, so "
                   "record cyclone rain is always under-forecast.")

    with st.expander("Bagging vs Boosting: Random Forest importance vs LightGBM SHAP"):
        comp = pd.DataFrame({"LightGBM mean |SHAP|": shap_imp / shap_imp.sum(), "Random Forest impurity": rf_imp / rf_imp.sum()}).fillna(0)
        comp = comp.loc[shap_imp.index[:12]][::-1]
        fig = go.Figure([go.Bar(y=comp.index, x=comp[c], name=c, orientation="h") for c in comp.columns])
        fig.update_layout(barmode="group", height=420, xaxis_title="share of total importance", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Both agree that persistence comes first. The RF spreads credit across correlated twins (rain_roll3, precipitation_hours), "
                   "while boosting uses pressure and the cross-city signals more.")

# ================================================================ TAB 4
def magnus_dewpoint(t, rh):
    a, b = 17.62, 243.12
    g = np.log(max(rh, 1) / 100) + a * t / (b + t)
    return b * g / (a - g)


def apply_tweaks(row, rain, hours, rh, dp, cloud):
    r = row.copy()
    d_rain = rain - row["rain_mm"]
    r["rain_mm"] = rain
    r[f"rain_today_{row['city'].lower()}"] = rain
    for w in (3, 7, 14, 30):
        r[f"rain_roll{w}"] = max(row[f"rain_roll{w}"] + d_rain / w, 0)
    if rain >= 1:
        r["wet_streak"], r["dry_streak"] = (row["wet_streak"] if row["rain_mm"] >= 1 else 1), 0
    else:
        r["wet_streak"], r["dry_streak"] = 0, (row["dry_streak"] if row["rain_mm"] < 1 else 1)
    r["precipitation_hours"] = hours
    if rh != row["relative_humidity_2m_mean"]:
        r["humidity_change_1d"] = row["humidity_change_1d"] + (rh - row["relative_humidity_2m_mean"])
        r["relative_humidity_2m_mean"] = rh
        # shift the real dew point by the Magnus-formula change, so untouched days keep their exact value
        r["dew_point_2m_mean"] = row["dew_point_2m_mean"] + magnus_dewpoint(row["temperature_2m_mean"], rh)             - magnus_dewpoint(row["temperature_2m_mean"], row["relative_humidity_2m_mean"])
        r["dewpoint_depression"] = row["temperature_2m_mean"] - r["dew_point_2m_mean"]
    r["pressure_msl_mean"] = row["pressure_msl_mean"] - row["pressure_change_1d"] + dp
    r["pressure_change_1d"] = dp
    r["cloud_cover_mean"] = cloud
    return r, d_rain


@st.cache_data
def heavy_rate_table():
    # empirical P(actual >= 10 mm | best-model forecast bin) on the test set
    bins = [0, 3, 6, 10, 15, 25, 1000]
    p = preds.assign(bin=pd.cut(preds[BEST], bins, right=False))
    return p.groupby("bin", observed=True)["rain_tomorrow_mm"].apply(lambda s: (s >= HEAVY).mean()), bins


with tab4:
    st.subheader("Pick a real day, change today's weather, and watch 9 models react")
    test_feat = feat[feat.split == "test"]
    c1, c2, c3 = st.columns([1, 1, 2])
    city = c1.selectbox("City", list(CITY_COLORS), key="pred_city")
    dates = test_feat.loc[test_feat.city == city, "date"].dt.date
    default_day = pd.Timestamp("2025-11-27").date()
    day = c2.date_input("'Today' (2024–2025)", value=default_day, min_value=dates.min(), max_value=dates.max())
    row = test_feat[(test_feat.city == city) & (test_feat.date.dt.date == day)]
    if row.empty:
        st.warning("No data for that day. Please pick another date.")
        st.stop()
    row = row.iloc[0]
    c3.markdown(f"<div class='note'><b>{city}, {day:%d %b %Y}</b> · {row['monsoon_season']}<br>"
                f"Real rain today: <b>{row['rain_mm']:.1f} mm</b> · humidity {row['relative_humidity_2m_mean']:.0f}% · "
                f"pressure change {row['pressure_change_1d']:+.1f} hPa</div>", unsafe_allow_html=True)

    st.markdown("**Today's weather (sliders start at the real values)**")
    s1, s2, s3, s4, s5 = st.columns(5)
    key = f"{city}-{day}"
    rain = s1.slider("Rain today (mm)", 0.0, 250.0, float(round(row["rain_mm"], 1)), 0.5, key=f"r{key}")
    hours = s2.slider("Hours of rain", 0.0, 24.0, float(row["precipitation_hours"]), 1.0, key=f"h{key}")
    rh = s3.slider("Humidity (%)", 50.0, 100.0, float(row["relative_humidity_2m_mean"]), 1.0, key=f"u{key}")
    dp = s4.slider("Pressure change vs yesterday (hPa)", -5.0, 5.0, float(np.clip(round(row["pressure_change_1d"], 1), -5, 5)), 0.1, key=f"p{key}")
    cloud = s5.slider("Cloud cover (%)", 0.0, 100.0, float(row["cloud_cover_mean"]), 1.0, key=f"c{key}")

    # sliders round the real values; snap back to the exact value when the user has not moved a slider
    rain = row["rain_mm"] if abs(rain - round(row["rain_mm"], 1)) < 1e-9 else rain
    dp = row["pressure_change_1d"] if abs(dp - np.clip(round(row["pressure_change_1d"], 1), -5, 5)) < 1e-9 else dp
    r, d_rain = apply_tweaks(row, rain, hours, rh, dp, cloud)
    X = pd.DataFrame([r[FEATURES]])
    for c in NUM:
        X[c] = X[c].astype(float)
    out = {}
    for name in ML_MODELS:
        out[name] = float(np.clip(np.expm1(models[name].predict(X)[0]), 0, None) * SMEAR[name])
    # ARIMA one-step forecast: adjust the saved real-history forecast for the changed rain today (phi1 + theta1) * delta
    base = preds[(preds.city == city) & (preds.date.dt.date == day)]["arima_next_log"].iloc[0]
    prm = arima_params[city]
    arima_log = base + (prm["ar.L1"] + prm["ma.L1"]) * (np.log1p(rain) - np.log1p(row["rain_mm"]))
    out["ARIMA (per city)"] = float(np.clip(np.expm1(arima_log), 0, None) * SMEAR["ARIMA (per city)"])
    Xh = X.assign(arima_next_log=arima_log)
    out["Hybrid RF + ARIMA"] = float(np.clip(np.expm1(arima_log + models["Hybrid RF + ARIMA"].predict(Xh)[0]), 0, None)
                                     * SMEAR["Hybrid RF + ARIMA"])
    out["Persistence"] = rain

    res = pd.DataFrame({"Model": out.keys(), "Forecast (mm)": out.values()})
    res["Family"] = res["Model"].map(test_tbl["Family"])
    res = res.sort_values("Forecast (mm)")
    best_mm = out[BEST]
    rates, bins = heavy_rate_table()
    b = pd.cut([best_mm], bins, right=False)[0]
    p_heavy = float(rates.get(b, np.nan))
    level, colr = (("HIGH", "#d62728") if best_mm >= HEAVY else ("MODERATE", "#e0a100") if best_mm >= 5 else ("LOW", "#2ca02c"))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{BEST.split(' (')[0]} forecast", f"{best_mm:.1f} mm")
    k2.metric("Stacking forecast", f"{out['Stacking (RF+LGBM+Cat+ANN)']:.1f} mm")
    k3.markdown(f"<div class='kpi'><div class='l'>Heavy-rain risk (≥10 mm)</div><div class='v' style='color:{colr}'>{level}</div>"
                f"<div class='l'>{p_heavy*100:.0f}% of test days with a similar forecast were heavy-rain days</div></div>",
                unsafe_allow_html=True)
    untouched = abs(d_rain) < 1e-9 and hours == row["precipitation_hours"] and rh == row["relative_humidity_2m_mean"] \
        and dp == row["pressure_change_1d"] and cloud == row["cloud_cover_mean"]
    if untouched:
        k4.metric("What actually fell tomorrow", f"{row['rain_tomorrow_mm']:.1f} mm")
    else:
        k4.metric("What actually fell tomorrow", "—", help="Hidden because you changed the weather: this is now a what-if scenario.")

    fig = px.bar(res, x="Forecast (mm)", y="Model", color="Family", orientation="h", color_discrete_map=FAMILY_COLORS,
                 text=res["Forecast (mm)"].round(1), labels={"Forecast (mm)": "Tomorrow's rain forecast (mm)", "Model": ""})
    fig.add_vline(x=HEAVY, line_dash="dot", line_color="#d62728", annotation_text="heavy rain 10 mm")
    if untouched:
        fig.add_vline(x=row["rain_tomorrow_mm"], line_color="black", annotation_text="actual", annotation_position="bottom right")
    fig.update_layout(height=470, margin=dict(t=70), yaxis=dict(categoryorder="array", categoryarray=res["Model"].tolist()),
                      legend=dict(orientation="h", y=1.04, yanchor="bottom", x=0, title_text=""))
    st.plotly_chart(fig, width="stretch")
    st.caption("All forecasts are in real mm (expm1 + smearing correction). Changing today's rain also updates the rolling means, wet/dry "
               "streaks, the city's cross-city rain feature and the ARIMA one-step forecast; humidity updates the dew point.")
    with st.expander("Try these real days"):
        st.markdown("""
- **Colombo, 27 Nov 2025 (default):** 111 mm fell today and Cyclone Ditwah brought 217 mm tomorrow. See how far every model under-forecasts the record.
- **Galle, a dry February day:** set humidity to 95% and pressure change to −2 hPa to watch the forecast jump.
- **Kandy in July (SW monsoon):** Kandy's sheltered location keeps forecasts low even with coastal rain.
""")

# ---------------------------------------------------------------- footer
st.divider()
with st.expander("🤖 GenAI Usage Declaration", expanded=False):
    st.markdown("""
In line with the module's Generative AI policy:
- **Claude (Anthropic)** was used as a coding assistant to write and scaffold the data download script, the notebook code
  (cleaning, feature engineering, model training, evaluation, SHAP) and **this dashboard's code and UI design**.
- **Gemini (Google)** was used to brainstorm the project plan.
- All AI-assisted code was **run, checked and reviewed by the author**. The dataset is **real** ERA5 reanalysis weather downloaded from
  Open-Meteo, not generated by AI, and every number shown comes from actually executing the notebook.
""")
st.caption(f"වැසි · University of Moratuwa, Semester 7 Machine Learning hands-on project · Sineth · "
           f"[GitHub]({REPO_URL}) · [Live app]({LIVE_URL}) · Weather data: Open-Meteo.com (CC BY 4.0), ERA5 by Copernicus / ECMWF")
