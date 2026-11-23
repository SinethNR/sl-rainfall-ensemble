# 🌧️ Wesi — Sri Lanka Rainfall Ensemble Lab

**Predicting tomorrow's rainfall in Colombo, Kandy and Galle with Bagging, Boosting, Stacking and an RF + ARIMA hybrid.**

> University of Moratuwa · Semester 7 · Machine Learning · Hands-on Project (Ensemble Learning Dashboard)
> Author: Sineth · Live dashboard: _link added after deployment_

---

## 1. The Sri Lankan problem

Sri Lanka's wet zone gets most of its rain from two monsoons: the **South-West monsoon (Yala, May–Sep)** and the **North-East monsoon (Maha, Dec–Feb)**, with heavy **inter-monsoon** storms in March–April and October–November. A single heavy day can flood Colombo's low-lying suburbs, trigger landslide warnings around Kandy, or wash out a fishing day in Galle. Paddy farmers, the Disaster Management Centre, and ordinary commuters all ask the same question every evening:

> **"How much rain will fall tomorrow?"**

This project answers that question with ensemble machine learning, using only real, verifiable Sri Lankan weather data, and explains **why** some ensembles beat others.

## 2. Dataset (real data, not synthetic)

| Item | Value |
|---|---|
| Source | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) |
| Underlying data | **ECMWF ERA5 / ERA5-Land reanalysis** (Copernicus Climate Change Service) |
| Cities | Colombo (6.93 N, 79.86 E), Kandy (7.29 N, 80.63 E), Galle (6.05 N, 80.22 E) |
| Period | 2015-01-01 → 2025-12-31 (11 years, daily) |
| Rows × columns | 12,054 × 22 (4,018 days × 3 cities) |
| Missing values | 0 |
| File | `data/sl_weather_2015_2025.csv` |
| Re-download | `python scripts/fetch_data.py` (no API key needed) |

**Exact API call (Colombo; Kandy and Galle only change latitude/longitude):**

```
https://archive-api.open-meteo.com/v1/archive?latitude=6.9271&longitude=79.8612&start_date=2015-01-01&end_date=2025-12-31&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,apparent_temperature_mean,precipitation_sum,rain_sum,precipitation_hours,relative_humidity_2m_mean,dew_point_2m_mean,pressure_msl_mean,cloud_cover_mean,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration&timezone=Asia%2FColombo
```

- Kandy: `latitude=7.2906&longitude=80.6337`
- Galle: `latitude=6.0535&longitude=80.2210`

### What "ERA5 reanalysis" means (and why it is real data)

ERA5 is produced by the European Centre for Medium-Range Weather Forecasts (ECMWF). It takes **millions of real observations** (weather stations, radiosonde balloons, aircraft, ships, buoys and satellites) and blends them with a physics-based weather model through **data assimilation**. The result is a consistent, gap-free hourly record on a ~9–31 km grid from 1940 to a few days ago.

- ✅ It is **observation-based**, not randomly generated or synthetic.
- ✅ It is **reproducible**: anyone running the URL above gets the same numbers.
- ⚠️ It is a **grid-cell estimate**, not a single rain gauge. It tends to smooth extreme downpours and over-count light drizzle days. We discuss this limitation in the notebook.

## 3. Machine learning task

- **Target:** `rain_tomorrow_mm`, the total precipitation (mm) for the next day in the same city.
- **Type:** Regression (with a rain / no-rain view for the dashboard).
- **Validation:** strict **time-based split** (train 2015–2022, validation 2023, test 2024–2025). No shuffling, so no future data leaks into training.

## 4. Models (Bagging · Boosting · Stacking · Hybrid)

| Family | Model | Idea |
|---|---|---|
| Baseline | Persistence ("tomorrow = today") and climatology | What we must beat |
| **Bagging** | Random Forest | Many deep trees on bootstrap samples; averaging lowers variance |
| **Boosting** | LightGBM, XGBoost, CatBoost | Trees built one after another, each fixing the previous errors; lowers bias |
| **Stacking** | RF + LightGBM + CatBoost + ANN (MLP) → Ridge meta-learner | A meta-model learns how much to trust each base model |
| **Hybrid** | ARIMA/SARIMA + Random Forest on residuals | ARIMA captures the linear seasonal signal, RF learns the non-linear left-overs |

Explainability: **SHAP** values and feature importance for the tree models.

## 5. Repository layout

```
Assignment 1/
├── data/
│   └── sl_weather_2015_2025.csv     # real ERA5 data from Open-Meteo
├── models/                          # trained models + metrics.json (created by the notebook)
├── scripts/
│   └── fetch_data.py                # re-downloads the dataset
├── ensemble_analysis.ipynb          # full executable notebook
├── app.py                           # Streamlit dashboard
├── requirements.txt
└── README.md
```

## 6. How to run

```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # optional: data is already in data/
jupyter notebook ensemble_analysis.ipynb   # run all cells -> writes models/
streamlit run app.py
```

## 7. Assignment requirement checklist

| Lecturer requirement | Where it is met |
|---|---|
| Ensemble Learning **dashboard** | `app.py` (Streamlit), deployed on Streamlit Community Cloud |
| **Bagging** | Random Forest (notebook Section 4, dashboard "Model Arena") |
| **Boosting** | LightGBM, XGBoost, CatBoost |
| **Stacking** | `StackingRegressor` with RF + LightGBM + CatBoost + ANN |
| Hybridize models (RF + ARIMA, RF + ANN) | Both: RF + ARIMA residual hybrid **and** RF + ANN inside the stack |
| **Sri Lankan dataset**, not US/Kaggle generic | Colombo, Kandy, Galle ERA5 weather (local weather/time-series, a suggested domain) |
| Python in an IDE / Colab | Jupyter notebook, VS Code |
| Compare model performance | Comparison table + charts (RMSE, MAE, R², rain/no-rain accuracy) |
| Interpret **why** models performed that way | Notebook Section 6 + dashboard "Why?" tab (bias–variance, SHAP, monsoon drivers) |
| Explicitly state AI-generated code | GenAI declaration below, at the top of the notebook, and in the dashboard footer |
| Executable `.ipynb` | `ensemble_analysis.ipynb`, runs top to bottom |
| Video ≤ 3 min with voice-over | 2:45 script provided separately |
| Live dashboard link (optional) | Streamlit Community Cloud URL above |

## 8. GenAI Usage Declaration

In line with the module's Generative AI policy, I declare that:

- **Claude (Anthropic)** was used as a coding assistant to write and scaffold the data download script, the notebook code (cleaning, feature engineering, model training, evaluation, SHAP analysis), the Streamlit dashboard code and UI design, and this README.
- **Gemini (Google)** was used to brainstorm the project plan and stage-by-stage workflow.
- All AI-generated code was **run, checked and reviewed by me**. The dataset is real (downloaded from Open-Meteo, not generated by AI), and every metric reported comes from actually executing the notebook.
- The project idea, the choice of the Sri Lankan rainfall problem, the review of results and the final interpretation are my own responsibility.

## 9. Credits

- Weather data: [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), based on ERA5 from the [Copernicus Climate Change Service](https://cds.climate.copernicus.eu/) / ECMWF.
- Hersbach, H. et al. (2020). *The ERA5 global reanalysis.* Quarterly Journal of the Royal Meteorological Society, 146(730), 1999–2049.
