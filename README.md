# ⚽ FIFA-Advanced-Player-Evaluation-System
> A full machine learning pipeline that predicts player **market value** (regression) and **performance tier** (classification) using diverse model architectures, hyperparameter tuning, and ensemble methods — benchmarked across two assignment iterations.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Key Results](#key-results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

---

## Overview

This project builds a dual-task machine learning system on FIFA player data:

**Task 1 — Regression:** Predict each player's market `Value Per M$` using continuous and categorical features.

**Task 2 — Classification:** Assign each player to one of four performance tiers (Low / Mid / High / Elite) derived from quantile-binning `Overall_Rating`.

The system spans two progressive assignments:

- **Assignment 2** establishes baselines using Polynomial Ridge Regression (R² ≈ 0.97) and Logistic Regression (Accuracy ≈ 81.7%)
- **Assignment 3** scales up with three diverse architectures (Random Forest, SVM, KNN), GridSearchCV tuning, learning curve analysis, and Stacking Ensembles — improving classification to over 86%

---

## Dataset

| Property | Details |
|---|---|
| **File** | `Fifa.csv` |
| **Domain** | Sports Analytics |
| **Target 1** | `Value Per M$` — continuous market value (Regression) |
| **Target 2** | `Performance_Tier` — 4-class tier label (Classification) |
| **Missing Values** | None |

### Features Used

| Feature | Type | Role |
|---|---|---|
| `Age` | Numeric | Predictor |
| `Overall_Rating` | Numeric | Strongest predictor (r=0.56 with value); **excluded from classification** to prevent data leakage |
| `Future Potential` | Numeric | Predictor |
| `Total_Stats Score` | Numeric | Predictor |
| `Position` | Categorical (OHE) | Predictor |
| `Country` | Categorical (OHE) | Predictor |
| `Name`, `Team` | Dropped | Non-predictive / high cardinality |

---

## Pipeline Architecture

```
Raw FIFA Data  (Fifa.csv)
       │
       ▼
┌──────────────────────────────────┐
│   1. EDA                         │  ← Distribution, correlation heatmap,
│                                  │    outlier detection, position analysis
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   2. Preprocessing               │  ← Drop Name/Team, train/test split (80/20),
│                                  │    IQR capping, OHE (Position, Country),
│                                  │    StandardScaler on numeric cols
└──────────────┬───────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
  REGRESSION        CLASSIFICATION
  Target:           Target:
  Value Per M$      Performance Tier
                    (quantile-binned
                     Overall_Rating)
       │                │
       ▼                ▼
┌──────────────────────────────────┐
│   Assignment 2 Baselines         │
│   Regression: Polynomial Ridge   │  ← deg=4, α≈19.3, Test R²≈0.97
│   Classification: LogReg + NB    │  ← LogReg wins at ~81.7% accuracy
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   Assignment 3 — Unified         │
│   Preprocessing Pipeline         │  ← IQRClipper + StandardScaler +
│   (sklearn Pipeline/             │    OneHotEncoder in ColumnTransformer
│    ColumnTransformer)            │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   3 Diverse Architectures        │
│   • Random Forest (tree-based)   │
│   • SVM / SVR (kernel-based)     │
│   • KNN (instance-based)         │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   GridSearchCV Tuning            │  ← 5-fold CV for RF & KNN
│   + Learning Curve Analysis      │    3-fold for SVM (cost)
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   Ensemble Methods               │
│   • Weighted Voting              │
│   • Stacking (Ridge / LogReg     │  ← Meta-learner
│     as meta-learner)             │
└──────────────┬───────────────────┘
               │
               ▼
    Unified Inference Pipeline
    Raw Player → Preprocessing → Stacking → Prediction
```

---

## Project Structure

```
fifa-player-evaluation/
│
├── notebook/
│   └── fifa_evaluation_pipeline.ipynb    ← Main notebook (Assignment 2 + 3)
│
├── data/
│   └── Fifa.csv
│
└── README.md
```

---

## Key Results

### Assignment 2 — Baselines

#### Regression (Polynomial Ridge)

| Degree | Test R² | Notes |
|---|---|---|
| 1 (Linear) | ~0.60 | Underfitting — non-linear data |
| 2 | ~0.85 | Improving |
| 3 | ~0.93 | Good |
| **4** | **~0.97** | **Best — smallest train/test gap** |

Best configuration: `degree=4`, `Ridge α≈19.3`

Lasso zeroed out **146 of 246 features** (59%) — mostly Country columns — confirming that player skill metrics drive market value, not nationality.

#### Classification (Logistic Regression vs Naïve Bayes)

| Model | Test Accuracy | Notes |
|---|---|---|
| Logistic Regression (L2) | ~81.7% | Best — handles OHE features well |
| GaussianNB | ~63% | Numerical only (3 features) |
| BernoulliNB | ~62% | Binary/presence features |
| ComplementNB | ~62% | Designed for text; mismatched domain |

Performance tiers (quantile-split, balanced at ~25% each):
`Low: ≤ rating threshold` | `Mid` | `High` | `Elite: top 25%`

---

### Assignment 3 — Advanced Models

#### Tuned Model Comparison

| Model | Regression R² | Classification Accuracy |
|---|---|---|
| Random Forest | Best individual reg | Best individual clf |
| SVM (RBF kernel) | Competitive | Competitive |
| KNN | Weakest of the three | Weakest of the three |

RBF kernel selected for both SVR and SVC after benchmarking linear, poly, and rbf across both tasks.

#### Ensemble Results

| System | Regression R² | Classification Accuracy |
|---|---|---|
| Assignment 2 (Poly Ridge / LogReg) | ~0.9746 | ~0.8167 |
| Assignment 3 Best Individual | ~0.9183 | ~0.8645 |
| Voting Ensemble | Weighted average | Soft vote |
| **Stacking Ensemble** | **Best overall** | **Best overall** |

Stacking used Ridge as meta-learner for regression and Logistic Regression for classification.

#### 5-Fold Cross-Validation Stability

All tuned models achieved **std < 0.05** across folds — confirming stable generalization.

---

## Technologies Used

| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Preprocessing | `sklearn.preprocessing` (StandardScaler, OneHotEncoder, PolynomialFeatures) |
| Pipelines | `sklearn.pipeline`, `sklearn.compose` (ColumnTransformer) |
| Regression Models | `LinearRegression`, `Ridge`, `Lasso`, `SVR`, `RandomForestRegressor`, `KNeighborsRegressor` |
| Classification Models | `LogisticRegression`, `GaussianNB`, `BernoulliNB`, `ComplementNB`, `SVC`, `RandomForestClassifier`, `KNeighborsClassifier` |
| Ensembles | `VotingRegressor`, `VotingClassifier`, `StackingRegressor`, `StackingClassifier` |
| Tuning & Validation | `GridSearchCV`, `KFold`, `StratifiedKFold`, `cross_val_score`, `learning_curve` |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fifa-player-evaluation.git
cd fifa-player-evaluation
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Add the dataset
Place `Fifa.csv` in the `data/` folder.

### 4. Run the notebook
```bash
jupyter notebook notebook/fifa_evaluation_pipeline.ipynb
```

> ⚠️ Run all cells **in order**. Assignment 3 cells depend on variables (`ASS2_REG_R2`, `ASS2_CLF_ACC`, `X_train_raw`, etc.) defined in Assignment 2 cells. Restarting mid-notebook will cause `NameError`.

---

## Authors

[Manahil Elhadi](https://github.com/mn05l)
Alexandria University, Faculty of Computers and Data Science
Machine Learning Course Project — 2025/2026
