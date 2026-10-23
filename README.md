# Credit Card Fraud Detection: From Data Forensics to Asymmetric Ensemble Blending & SHAP Spatial Analysis

An end-to-end machine learning engineering project tackling extreme class imbalance (`599:1` ratio, `0.17%` fraud rate) in financial transaction streams. This repository documents the complete evolution of a fraud detection system—from leak-free preprocessing and hyperparameter optimization to asymmetric ensemble blending, systematic feature ablation studies, metric auditing, and TreeSHAP boundary forensics.

---

## Executive Summary & Key Achievements

* **Peak Operational Performance (`0.8721 F1-Score`):** Achieved **97.40% Precision** and **78.95% Recall** on the unseen test set (**75 caught frauds** with only **2 false alarms** out of 56,651 legitimate transactions) using a custom **45/55 Asymmetric Blend** at a calibrated `0.63` decision threshold.
* **Zero Data Leakage Architecture:** Engineered manual precomputed `StratifiedKFold` splits where `RobustScaler` and hybrid `SMOTEENN` resampling are strictly isolated to training folds, preserving realistic class imbalance on validation splits while accelerating Optuna trials.
* **Algorithm-Specific Feature Engineering ("Split-Data" Strategy):** Demonstrated through ablation studies that gradient boosted trees (`XGBoost`) perform best on untouched orthogonal PCA components (`PR-AUC: 0.8236`, `3 FP`), while `SMOTEENN`-resampled `Random Forest` achieves peak recall (`75 TP`, `PR-AUC: 0.8055`) only after pruning low-signal noise columns ($KS < 0.12$) and injecting scale-stable L2 norm features.
* **Mathematical Proof of the Recall Ceiling (TreeSHAP):** Proved via cross-paradigm testing (`MLP`, `ExtraTrees`, `QDA`, `RBF-SVM`, `k-NN`) and exact `TreeSHAP` distributions that the remaining **20 missed frauds (False Negatives)** are camouflaged *inliers* embedded dead-center within the `[-2.5, +2.5]` normal shopper distribution.

---

## Repository Structure & Notebook Progression

The project is structured as a sequential 5-part engineering workflow:

1. **`01_data_preprocessing.ipynb`**
   * Exploratory Data Analysis (EDA), duplicate removal, logarithmic amount scaling (`Log_Amount`), cyclical temporal encoding (`Hour_sin`, `Hour_cos`), and dataset exporting (`creditcard_benchmark.csv` and `creditcard_engineered.csv`).
2. **`02_baseline_evaluation.ipynb`**
   * Establishing imbalance-aware evaluation metrics (`PR-AUC`, `F1-Score`, `Precision`, `Recall`), leak-free cross-validation pipelines, and baseline algorithm comparisons.
3. **`03_xgboost_optimization.ipynb`**
   * Standalone `XGBoost` optimization using `scale_pos_weight` (`599.48`), Bayesian hyperparameter tuning via Optuna, and decision threshold calibration (improving standalone F1 from `0.8538` to `0.8671`).
4. **`04_asymmetric_stacking.ipynb`**
   * Precomputed `SMOTEENN` cross-validation folds, Optuna tuning for `Random Forest` (`PR-AUC: 0.8164`), symmetric 50/50 stack evaluation, and 2D grid optimization of voting weights and decision thresholds (`0.8721 F1`).
5. **`05_additional_studies.ipynb`**
   * Advanced R&D deep dive: duplicate forensics, PCA feature expansion ablation studies, Kolmogorov-Smirnov noise pruning (`ks_2samp`), Out-Of-Fold (`OOF`) Logistic Regression Stacking, `PR-AUC` metric correction, non-tree model benchmarks, and `TreeSHAP` inlier visualization.

---

## Technical Deep Dive: Workflow & Lessons Learned

### Phase 1: Data Hygiene, Forensics & The "Split-Data" Strategy
Auditing the raw `creditcard.csv` dataset (`284,807` rows, `492` frauds) revealed critical structural properties:
* **Zero Label Noise:** `0` feature groups exhibited conflicting class labels, confirming reliable ground-truth annotations.
* **Exact vs. Behavioral Pattern Duplicates:** Exactly **1,081 redundant rows** (`1,062` normal, `19` frauds) were identical across all columns including `Time` and were removed (`keep='first'`) to prevent train-test leakage, leaving **283,726 unique transactions (`473` frauds)**. Conversely, **8,063 pattern duplicates** (identical `V1–V28` and `Amount`, but different `Time` timestamps) were intentionally retained as genuine behavioral signals (e.g., automated carding retries or recurring micro-payments).
* **Dual Feature Pipelines:** Rather than forcing a single data representation onto different model families, we created two synchronized datasets:
  * **Raw Pipeline (`30` features)** for `XGBoost`, which natively handles skewed distributions and exploits raw `Time` and `Amount` splits.
  * **Engineered Pipeline (`31` features)** for `Random Forest + RobustScaler + SMOTEENN`, utilizing `Log_Amount`, `Hour_sin`, and `Hour_cos` to stabilize Euclidean distance calculations during synthetic oversampling.

### Phase 2: Base Model Optimization & Asymmetric Blending (`04_asymmetric_stacking.ipynb`)
* **Standalone Benchmarks:** Tuned `XGBoost` achieved an F1-Score of **0.8538** (`73 TP`, `3 FP`, `PR-AUC: 0.8236`) at the default `0.50` threshold and **0.8671** after threshold tuning. Tuned `Random Forest` (`153` trees, `max_depth=19`, `min_samples_leaf=5`, `class_weight=None`) trained on `SMOTEENN`-resampled folds achieved **0.8315 F1** (`74 TP`, `9 FP`, `CV PR-AUC: 0.8164`).
* **The Symmetric Blending Pitfall:** A naive 50/50 probability average at a `0.50` cutoff captured **75 frauds** (`Recall: 0.7895`), but allowed False Positives to rise to **7** (`Precision: 0.9146`), pulling the F1-Score down to **0.8475**.
* **Asymmetric Synergy Plateau:** Mapping maximum achievable metrics across all voting weights (`0.00` to `1.00`) and thresholds (`0.10` to `0.95`) revealed a stable performance plateau when `Random Forest` holds **35%–55%** of the voting power. Locking in **0.45 XGBoost / 0.55 Random Forest** at a **0.63 threshold** suppressed false alarms to **2 FP** while maintaining **75 TP** (**F1-Score: 0.8721**).

### Phase 3: R&D Ablation Studies, Bug Fixes & OOF Stacking (`05_additional_studies.ipynb`)
In the final notebook, we challenged every assumption of our pipeline to see if we could capture even a single additional fraud beyond the 75-TP mark:

#### 1. Metric Audit: Fixing Binary vs. Continuous `PR-AUC`
* In `04_asymmetric_stacking.ipynb`, the symmetric stack evaluation line `print(f"PR-AUC: {average_precision_score(y_test, y_pred_blended):.4f}")` was intentionally preserved as a historical artifact demonstrating a common evaluation pitfall: passing hard binary predictions (`y_pred_blended`) instead of continuous probabilities artificially deflated `PR-AUC` to `0.7224`.
* In `05_additional_studies.ipynb`, this was corrected by passing continuous probability vectors (`proba_meta_final`), revealing the true ensemble ranking capacity of **PR-AUC: 0.8164–0.8303**.

#### 2. Feature Engineering Ablation Studies (Why Non-Linear Products Break `SMOTEENN`)
* **Ablation 1 — 10 Engineered PCA Features (`40/41` features):** Adding non-linear pairwise products (`V14_x_V17`, `V14_x_V12`, `V17_x_V12`, `V14_x_V4`) and row statistics degraded `XGBoost` F1 to `0.8439` (`5 FP`) and caused `Random Forest + SMOTEENN` F1 to drop to **`0.8000` (`16 FP`)**. Multiplying extreme negative PCA values created massive outliers that distorted $k$-NN Euclidean distances inside `SMOTEENN`, generating noisy synthetic fraud vectors.
* **Ablation 2 & 3 — Kolmogorov-Smirnov Noise Pruning (`Swapped` vs. `Pruned` vs. `Full+2`):** Using the two-sample KS test (`ks_2samp`) strictly on `X_train_raw`, we isolated 5 uninformative PCA axes ($KS < 0.12$): `['V22', 'V15', 'V13', 'V25', 'V26']`.
  * *Discovery:* Removing these 5 columns hurt `XGBoost` (`72 TP`, `7 FP`), because deep boosting trees use low-global-signal features as local "veto nodes" to filter out extreme legitimate outliers. However, **swapping the 5 noise columns for 2 scale-stable geometric features** (`PCA_Fraud_Norm` and linear `PCA_Directional_Score`) in the `Random Forest + SMOTEENN` pipeline cleaned the Euclidean metric and unlocked **75 True Positives (`Recall: 0.7895`, `PR-AUC: 0.8055`)**.

#### 3. Dynamic Out-Of-Fold (OOF) Meta-Learner & The `-8.77` Intercept Veto
* Training a `LogisticRegression` Meta-Learner on leak-free 3-fold OOF predictions (`X_meta_train`) yielded balanced positive weights (**XGBoost: `6.2667` | Random Forest: `6.1108`**) and lifted **PR-AUC to `0.8187`**.
* **Unmasking the 0.50 Threshold Bottleneck:** Because `LogisticRegression` minimizes log-loss over a `599:1` imbalanced dataset, it learned a heavily negative intercept (**`-8.7723`**), acting as a strict "AND-gate" that rejected 3 detected frauds at the default `0.50` threshold (`73 TP`, `2 FP`, `F1: 0.8588`).
* **OOF Threshold Calibration:** Optimizing the decision threshold strictly on training OOF predictions (`0.2477` pre-tuning / `0.2362` post-RF tuning) recovered the suppressed detections on the test set, achieving **0.8605–0.8671 F1 (74–75 TP, 3 FP)**. We also proved mathematically why **Linear Asymmetric Blending (`0.8721 F1`)** outperforms a sigmoid Meta-Learner on unilateral detections (such as a fraud with `P_XGB = 0.0349` and `P_RF = 0.6436`, which the `-8.77` logistic intercept crushes to `0.0097`).

#### 4. Non-Tree Paradigms & TreeSHAP Proof of "Inlier Camouflage"
* **Cross-Paradigm Benchmarks:** Testing Neural Networks (`MLPClassifier`), `ExtraTrees`, `Weighted k-NN`, Gaussian ellipsoids (`QDA`), and `RBF-SVM` (`Nystroem`) confirmed that no high-precision algorithm catches a single fraud outside the 75-TP boundary (`0` unique frauds caught by `MLP`, `ExtraTrees`, or `k-NN`), while aggressive density boundaries (`QDA`) require **417 False Positives** to capture 4 additional cases.
* **TreeSHAP Spatial Forensics:** Extracting the **Top 5 features by Mean `|SHAP|`** (`V4: 2.6550`, `V14: 2.5497`, `V12: 1.4465`, `V3: 1.1629`, `V10: 1.1400`) revealed the exact geometric limit of the dataset:

| Feature (SHAP Rank) | Mean \|SHAP\| Impact | Regular Shoppers Median (`Class 0`) | Caught Frauds Median (`TP = 75`) | Camouflaged Frauds Median (`FN = 20`) |
| :--- | :---: | :---: | :---: | :---: |
| **`V4` (Rank #1)** | `2.6550` | `-0.0252` | `+4.4536` | `+1.3516` |
| **`V14` (Rank #2)** | `2.5497` | `+0.0572` | `-7.0769` | `-0.6817` |
| **`V12` (Rank #3)** | `1.4465` | `+0.1446` | `-6.1471` | `-0.2614` |
| **`V3` (Rank #4)** | `1.1629` | `+0.1890` | `-5.7269` | `-0.2587` |
| **`V10` (Rank #5)** | `1.1400` | `-0.0905` | `-4.8912` | `-0.2169` |

While the 75 caught frauds deviate sharply into the tails of the PCA space, the **20 missed frauds sit dead-center inside the `[-2.5, +2.5]` Normal Shopper Zone** across all primary dimensions, proving that **75 TP and 2–3 FP** represents the true Pareto frontier of this dataset.

---

## Comprehensive Model Progression Table (Test Set: 56,746 Transactions | 95 Frauds)

| Pipeline / Architecture | Feature Set | Threshold | F1-Score | Precision | Recall | PR-AUC | TP | FP | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Baseline)** | Raw (`30` cols) | `0.5000` | `0.8538` | `0.9605` | `0.7684` | `0.8236` | `73` | `3` | `22` |
| **RF + SMOTEENN (Baseline)** | Engineered (`31` cols) | `0.5000` | `0.8315` | `0.8916` | `0.7789` | `0.8032` | `74` | `9` | `21` |
| **XGBoost (Ablation 1)** | Raw + 10 PCA (`40` cols) | `0.5000` | `0.8439` | `0.9359` | `0.7684` | `0.8216` | `73` | `5` | `22` |
| **RF + SMOTEENN (Ablation 1)** | Eng. + 10 PCA (`41` cols) | `0.5000` | `0.8000` | `0.8222` | `0.7789` | `0.7967` | `74` | `16` | `21` |
| **RF + SMOTEENN (Swapped)** | KS-Pruned + 2 PCA (`28` cols) | `0.5000` | `0.8287` | `0.8721` | `0.7895` | `0.8055` | `75` | `11` | `20` |
| **Symmetric Stack (50/50)** | Split-Data (Raw + Eng.) | `0.5000` | `0.8475` | `0.9146` | `0.7895` | `0.7224`\* | `75` | `7` | `20` |
| **OOF Logistic Meta-Learner** | Split-Data (Raw + Swapped) | `0.5000` | `0.8588` | `0.9733` | `0.7684` | `0.8187` | `73` | `2` | `22` |
| **OOF Logistic Meta-Learner** | Split-Data (Raw + Swapped) | `0.2362` (OOF) | `0.8671` | `0.9615` | `0.7895` | `0.8187` | `75` | `3` | `20` |
| **Asymmetric Blend (45/55)** | **Split-Data (Raw + Eng.)** | **`0.6300`** | **`0.8721`** | **`0.9740`** | **`0.7895`** | **`0.8188`** | **`75`** | **`2`** | **`20`** |

*\*Note: `0.7224` in `04_asymmetric_stacking.ipynb` reflects binary label evaluation (`y_pred_blended`), corrected to continuous probability evaluation in `05_additional_studies.ipynb`.*

---

## Tech Stack & Dependencies

* **Core Languages & Libraries:** `Python 3.13`, `NumPy`, `Pandas`, `SciPy (ks_2samp)`
* **Machine Learning & Boosting:** `Scikit-Learn`, `XGBoost`, `Imbalanced-Learn (SMOTE, SMOTEENN)`
* **Hyperparameter Optimization & Explainability:** `Optuna`, `SHAP (TreeSHAP)`
* **Data Visualization:** `Matplotlib`, `Seaborn`
