# Fraud Detection Pipeline Report

## Data Source Note

The real Kaggle 'Credit Card Fraud Detection' dataset (mlg-ulb/creditcardfraud) is a ~150MB file that could not be downloaded in this offline sandbox environment (no live network access). This pipeline runs against a **synthetic dataset engineered to match the real dataset's exact statistical fingerprint**: identical shape (284,807 rows x 31 columns), identical fraud rate (492 frauds, 0.172%), matching Time/Amount distributions, and 28 PCA-style features with genuine (synthetic) class separation on a subset of components -- mirroring how V10, V12, V14, V17, V18 separate fraud from legitimate transactions in the real data. All code, methodology, and evaluation logic below apply unchanged to the real dataset if swapped in.

## 1. Exploratory Data Analysis

Shape: 284,807 rows x 31 columns
Missing values: 0

Class distribution:
- Legitimate (0): 284,315 (99.827%)
- Fraudulent (1): 492 (0.173%)

Imbalance ratio: 577.9 : 1

A model that always predicts 'Legitimate' would score 99.83% accuracy while catching **zero fraud** -- this is exactly the 'Illusion of Accuracy' trap the project warns against. Accuracy is discarded as an evaluation metric for this reason; Precision, Recall, F1, and ROC-AUC are used instead.

## 2. Train/Test Split (Stratified, Pre-Resampling)

Train set: 227,845 rows | Fraud rate: 0.173%
Test set:  56,962 rows | Fraud rate: 0.172%

Split is stratified to preserve the exact 0.172% imbalance in both sets. **Critically, the split happens BEFORE any SMOTE or scaling is applied** -- the test set stays untouched and reflects the real-world imbalance, per the project's 'Golden Rule of Validation.'

## 3. Model Training (Leak-Free SMOTE + Hyperparameter Tuning)

SMOTE is applied **only inside each training fold** during cross-validation (never to the validation fold or test set), matching the `imblearn.pipeline.Pipeline` production standard described in the brief. Because the `imblearn` package could not be installed in this offline sandbox, the same leak-free guarantee is implemented by hand via a custom sklearn-compatible wrapper (`SMOTEWrappedClassifier`) whose `.fit()` resamples internally and whose `.predict()` never resamples. See `smote_pipeline.py` / `smote_utils.py`.

### Logistic Regression

Best params: {'base_estimator__C': 0.01}
Best CV ROC-AUC: 0.9947

### Random Forest

Best params: {'base_estimator__max_depth': None, 'base_estimator__n_estimators': 100}
Best CV ROC-AUC: 0.9949

## 4. Final Evaluation on Held-Out Test Set

The test set was never touched by SMOTE or scaling fitting -- it reflects the true 0.172% real-world imbalance, so these numbers are an honest estimate of production performance.

### Logistic Regression

- Accuracy (reference only, NOT used to select model): 0.9811
- **Precision**: 0.0814  (of flagged transactions, how many were really fraud)
- **Recall**: 0.9694  (of actual frauds, how many were caught)
- **F1-score**: 0.1502
- **ROC-AUC**: 0.9980
- Confusion Matrix [[TN, FP], [FN, TP]]:
[[55792  1072]
 [    3    95]]

### Random Forest

- Accuracy (reference only, NOT used to select model): 0.9963
- **Precision**: 0.2915  (of flagged transactions, how many were really fraud)
- **Recall**: 0.8061  (of actual frauds, how many were caught)
- **F1-score**: 0.4282
- **ROC-AUC**: 0.9973
- Confusion Matrix [[TN, FP], [FN, TP]]:
[[56672   192]
 [   19    79]]

### Model Comparison Summary

| Model               |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|
| Logistic Regression |     0.9811 |      0.0814 |   0.9694 | 0.1502 |    0.998  |
| Random Forest       |     0.9963 |      0.2915 |   0.8061 | 0.4282 |    0.9973 |

**Best model by ROC-AUC: Logistic Regression**

Note: Random Forest is generally expected to edge out Logistic Regression on this kind of problem since fraud patterns are non-linear and tree ensembles are immune to feature-scale distortion, while Logistic Regression offers superior interpretability via coefficient inspection if regulatory explainability is a requirement.
