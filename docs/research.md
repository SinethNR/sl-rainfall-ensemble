## 9. Research: comparing ensemble methods, and where they work (tabular vs image/text data)

This section answers the lecture's research questions: *RF vs gradient boosting (XGBoost, LightGBM, CatBoost)*, and *strengths and limitations on tabular vs image/text data*.

### 9.1 Random Forest vs gradient boosting: what our experiment shows

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

### 9.2 Tabular vs image / text data: strengths and limitations

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
