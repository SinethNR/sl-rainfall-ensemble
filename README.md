# 🌧️ වැසි (Wæsi, "rain"): Sri Lanka Rainfall Ensemble Lab

**Predicting tomorrow's rainfall in Colombo, Kandy and Galle with Bagging, Boosting, Stacking and an RF + ARIMA hybrid.**

> University of Moratuwa · Semester 7 · Machine Learning · Hands-on Project (Ensemble Learning Dashboard)
> Author: Sineth · **Live dashboard: https://wesi-sl-rainfall.streamlit.app**

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

## 7. Dashboard (`app.py`)

| Tab | What it shows |
|---|---|
| 🌦️ Overview & climate | KPI cards, two-peak monsoon climatology, city map, season × city heavy-rain heatmap, annual totals, Cyclone Roanu / Ditwah callout |
| 🏆 Model arena | Filterable test leaderboard, metric bar charts, paired-bootstrap significance, smearing before/after, forecast vs reality time series, error by city and season |
| 🔍 Why? Explainability | SHAP importance + beeswarm, why XGBoost led, why stacking flattened cyclone peaks, how RF + ARIMA beat ARIMA, stacking weights, extreme-day error share |
| 🔮 Live predictor | Pick a real 2024–2025 day and city, move today's weather sliders, and see all 9 models' forecasts in mm with a heavy-rain risk level |

The app loads the saved artefacts in `models/`, so it starts in seconds and never retrains.

**Deploy (Streamlit Community Cloud):** share.streamlit.io → *Create app* → repo `SinethNR/sl-rainfall-ensemble`, branch `main`, file `app.py` → *Advanced settings* → Python **3.11** → *Deploy*.

## 8. Final test results (2024–2025, metrics in mm)

| Model | Family | RMSE | MAE | R² | Heavy-rain AUC |
|---|---|---|---|---|---|
| **XGBoost** | Boosting | **9.75** | 5.46 | **0.346** | 0.824 |
| Random Forest | Bagging | 9.92 | **5.42** | 0.324 | 0.823 |
| LightGBM | Boosting | 9.92 | 5.51 | 0.324 | 0.823 |
| Stacking (RF + LGBM + Cat + ANN) | Stacking | 9.95 | 5.47 | 0.320 | **0.825** |
| CatBoost | Boosting | 9.98 | 5.43 | 0.316 | 0.824 |
| Hybrid RF + ARIMA | Hybrid | 10.02 | 5.46 | 0.309 | 0.819 |
| Ridge (linear) | Baseline | 10.19 | 5.71 | 0.287 | 0.817 |
| ARIMA (per city) | Time-series | 10.27 | 5.58 | 0.276 | 0.802 |
| ANN (MLP) | Neural net | 10.56 | 5.91 | 0.234 | 0.813 |
| Climatology | Baseline | 11.59 | 6.54 | 0.077 | 0.667 |
| Persistence | Baseline | 11.90 | 6.17 | 0.026 | 0.803 |

The top ensembles are statistically tied (paired bootstrap). See notebook Section 8 for the full "why" analysis.

## Research: comparing ensemble methods, and where they work (tabular vs image/text data)

This section answers the lecture's research questions: *RF vs gradient boosting (XGBoost, LightGBM, CatBoost)*, and *strengths and limitations on tabular vs image/text data*.

### R.1 Random Forest vs gradient boosting: what our experiment shows

| Aspect | Random Forest (bagging) | XGBoost / LightGBM / CatBoost (boosting) | Evidence in this project |
|---|---|---|---|
| How trees are built | Independently, in parallel, on bootstrap samples | Sequentially; each tree fits the previous errors | Section 4 |
| Main effect | Lowers **variance** | Lowers **bias** | RF best MAE (5.42 mm); XGBoost best RMSE/R² (9.75 mm, 0.346) |
| Tuning effort | Low: works well with defaults | Higher: learning rate, depth, trees; needs early stopping | Boosters needed a validation year to choose 173–290 trees |
| Over-fitting risk | Low (averaging) | Higher without early stopping or regularisation | Early stopping on 2023 |
| Speed / size | Slower to predict, large files (6 MB) | Fast, compact models (0.2 MB) | `models/` sizes |
| Extremes | Averages leaves, so it caps forecasts (max 52 mm) | Reaches further into the tail (max 66 mm) | Cyclone Ditwah analysis, Section 5.0b |
| Categorical data | Needs one-hot encoding | CatBoost handles categories natively; LightGBM / XGBoost support them too | `city`, `monsoon_season` |

**Verdict on our data:** the top tree ensembles are statistically tied (paired bootstrap). Boosting wins on the error metric that punishes big misses (RMSE), bagging wins on typical-day error (MAE), and stacking wins on ranking heavy-rain days (AUC).

### R.2 Tabular vs image / text data: strengths and limitations

| | Tabular data (our rainfall table, telecom churn, house prices) | Image / text / audio data |
|---|---|---|
| Typical best models | **Tree ensembles** (RF, XGBoost, LightGBM, CatBoost) usually match or beat deep learning (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022) | **Deep neural networks** (CNNs, Vision Transformers, Transformers / LLMs) |
| Why | Features are already meaningful (humidity, pressure); trees split on thresholds, handle mixed types, missing values and irregular, non-smooth patterns; little data needed | Meaning lives in **spatial or sequential structure** (pixels next to pixels, words in context). Trees cannot learn convolution or attention-style feature extraction from raw inputs |
| Ensemble strengths | Fast, explainable (SHAP, importance), robust with ~10k rows, little tuning | Ensembles of networks (snapshot ensembles, model averaging, test-time augmentation) still help, but each member is expensive |
| Limitations | No extrapolation beyond the training range (our cyclone peaks); feature engineering matters a lot; time order must be respected | Trees on raw pixels or bag-of-words lose structure and scale poorly to very high dimensions; they need hand-made features first (e.g. HOG, TF-IDF, or CNN / BERT embeddings) |
| Practical hybrid | RF + ANN or RF + ARIMA (Sections 4.5–4.6) | **Deep features + boosting**: extract embeddings with a CNN or transformer, then train XGBoost / LightGBM on top |
| Sri Lankan example | Rainfall, telecom churn, ikman.lk prices | Satellite cloud images for monsoon nowcasting, Sinhala / Tamil text sentiment, X-ray images |

**Takeaway:** ensemble trees are the right tool for this problem because weather records are tabular. If we added **satellite cloud imagery**, a CNN would be needed to read it, and its output could then be stacked with the tree ensembles. That is a natural next step for this project and a possible research paper.

*References:* Breiman, L. (2001) *Random Forests*, Machine Learning 45; Chen & Guestrin (2016) *XGBoost*, KDD; Ke et al. (2017) *LightGBM*, NeurIPS; Prokhorenkova et al. (2018) *CatBoost*, NeurIPS; Zhang, G.P. (2003) *Time series forecasting using a hybrid ARIMA and neural network model*, Neurocomputing 50; Grinsztajn, Oyallon & Varoquaux (2022) *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS; Shwartz-Ziv & Armon (2022) *Tabular data: Deep learning is not all you need*, Information Fusion 81; Hersbach et al. (2020) *The ERA5 global reanalysis*, QJRMS 146.


## 9. Assignment requirement checklist

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
| Research: RF vs boosting, tabular vs image/text | README *Research* section, notebook Section 9, dashboard "Why?" tab |
| Interpret **why** models performed that way | Notebook Section 6 + dashboard "Why?" tab (bias–variance, SHAP, monsoon drivers) |
| Explicitly state AI-generated code | GenAI declaration below, at the top of the notebook, and in the dashboard footer |
| Executable `.ipynb` | `ensemble_analysis.ipynb`, runs top to bottom |
| Video ≤ 3 min with voice-over | 2:45 script provided separately |
| Live dashboard link (optional) | https://wesi-sl-rainfall.streamlit.app |

## 10. GenAI Usage Declaration

In line with the module's Generative AI policy, I declare that:

- **Claude (Anthropic)** was used as a coding assistant to write and scaffold the data download script, the notebook code (cleaning, feature engineering, model training, evaluation, SHAP analysis), the Streamlit dashboard code and UI design, and this README.
- **Gemini (Google)** was used to brainstorm the project plan and stage-by-stage workflow.
- All AI-generated code was **run, checked and reviewed by me**. The dataset is real (downloaded from Open-Meteo, not generated by AI), and every metric reported comes from actually executing the notebook.
- The project idea, the choice of the Sri Lankan rainfall problem, the review of results and the final interpretation are my own responsibility.

## 11. Credits

- Weather data: [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), based on ERA5 from the [Copernicus Climate Change Service](https://cds.climate.copernicus.eu/) / ECMWF.
- Hersbach, H. et al. (2020). *The ERA5 global reanalysis.* Quarterly Journal of the Royal Meteorological Society, 146(730), 1999–2049.
