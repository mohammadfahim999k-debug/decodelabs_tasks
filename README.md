# Project 2: Supervised Learning — Fraud Detection Pipeline

A leak-free, imbalanced-classification pipeline that detects fraudulent
credit card transactions, built as part of the DecodeLabs Data Science
Industrial Training Kit (Batch 2026).

## Overview

This project builds and tunes classification models to catch fraud in a
dataset where only 0.17% of transactions are actually fraudulent (577.9:1
imbalance). The focus is not just "getting a model to run" but doing it
**correctly**: no data leakage, no misleading accuracy metrics, and an honest
precision/recall tradeoff analysis.

## ⚠️ Data Source Note

The real Kaggle "Credit Card Fraud Detection" dataset
(`mlg-ulb/creditcardfraud`, ~150MB) could not be downloaded in the original
offline development sandbox used to build this project (no live network
access). Instead, `generate_dataset.py` produces a **synthetic dataset
engineered to match the real dataset's exact statistical fingerprint**:

- Identical shape: 284,807 rows × 31 columns
- Identical fraud count: 492 frauds (0.172% fraud rate)
- Matching `Time` / `Amount` distributions
- 28 PCA-style anonymized features with realistic, overlapping (not
  artificially clean) class separation

All pipeline code and methodology apply unchanged if the real Kaggle CSV is
substituted in — only the input file changes.

## Repository Contents

| File | Description |
|---|---|
| `generate_dataset.py` | Regenerates the full 284,807-row synthetic dataset (run this first) |
| `creditcard_sample.csv` | First 500 rows, for quick preview without regenerating the full file |
| `pipeline.py` | Main pipeline: EDA → leak-free split → SMOTE + GridSearchCV → evaluation |
| `smote_utils.py` | Manual, vectorized SMOTE implementation (Chawla et al., 2002) |
| `smote_pipeline.py` | Leak-free scikit-learn-compatible SMOTE-wrapped classifier |
| `fraud_report.md` | Auto-generated markdown report from the pipeline run |
| `Project2_Fraud_Detection_Report.docx` | Full write-up with charts and analysis |
| `*.png` | Evaluation charts (class imbalance, ROC curves, confusion matrices, model comparison) |

## What Was Done

**1. Confirmed the "Illusion of Accuracy" trap**
A model predicting "Legitimate" for every transaction scores 99.83% accuracy
while catching zero fraud. Accuracy is used only as a reference figure — never
to select or tune a model.

**2. Leak-free train/test split**
Stratified 80/20 split *before* any resampling or scaling, preserving the
true 0.172% imbalance in both sets.

**3. SMOTE implemented from scratch**
`imblearn` wasn't installable offline, so SMOTE (k=5 nearest-neighbor
interpolation) was implemented manually and wrapped in a custom
`SMOTEWrappedClassifier` whose `.fit()` resamples internally and whose
`.predict()` never does — reproducing `imblearn.pipeline.Pipeline`'s
leak-free guarantee without the dependency. A majority-class undersampling
cap (20,000 rows) was added as a standard hybrid-resampling technique for
computational tractability.

**4. Two models trained via GridSearchCV**
Logistic Regression (with `StandardScaler`, fit train-fold-only) and Random
Forest, tuned via 3-fold stratified cross-validation scored on ROC-AUC.

**5. Evaluated on an untouched test set**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.081 | 0.969 | 0.150 | 0.998 |
| Random Forest | 0.292 | 0.806 | 0.428 | 0.997 |

Logistic Regression catches nearly all fraud but with many false alarms;
Random Forest is more balanced. Which is "better" depends on the real-world
cost of a missed fraud vs. a false decline — see the full report for the
tradeoff discussion.

## Tech Stack
Python · Pandas · NumPy · scikit-learn · Matplotlib

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
python generate_dataset.py    # creates creditcard.csv (~160MB)
python pipeline.py            # runs the full pipeline, writes fraud_report.md
```

## Full Report

See `Project2_Fraud_Detection_Report.docx` for the complete write-up,
including the precision/recall tradeoff discussion and all evaluation
charts.
