"""
================================================================================
 Project 2: Supervised Learning — Fraud Detection Pipeline | DecodeLabs
================================================================================
 Input : creditcard.csv (284,807 transactions, 0.172% fraud rate)
 Output: fraud_report.md, trained model artifacts, evaluation charts

 Pipeline stages (per project brief):
   1. EDA — confirm extreme class imbalance
   2. Stratified train/test split BEFORE any resampling or scaling
   3. Leak-free SMOTE (fit only inside each training fold)
   4. Train Logistic Regression + Random Forest via GridSearchCV
   5. Evaluate with Precision, Recall, F1, ROC-AUC, confusion matrix
      -- accuracy is explicitly NOT used as a decision metric
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, accuracy_score
)

from smote_pipeline import SMOTEWrappedClassifier

RANDOM_STATE = 42
pd.set_option("display.width", 120)

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

log("# Fraud Detection Pipeline Report\n")

# ------------------------------------------------------------------
# NOTE ON DATA SOURCE
# ------------------------------------------------------------------
log("## Data Source Note\n")
log("The real Kaggle 'Credit Card Fraud Detection' dataset (mlg-ulb/creditcardfraud) "
    "is a ~150MB file that could not be downloaded in this offline sandbox "
    "environment (no live network access). This pipeline runs against a "
    "**synthetic dataset engineered to match the real dataset's exact statistical "
    "fingerprint**: identical shape (284,807 rows x 31 columns), identical fraud "
    "rate (492 frauds, 0.172%), matching Time/Amount distributions, and 28 "
    "PCA-style features with genuine (synthetic) class separation on a subset "
    "of components -- mirroring how V10, V12, V14, V17, V18 separate fraud from "
    "legitimate transactions in the real data. All code, methodology, and "
    "evaluation logic below apply unchanged to the real dataset if swapped in.\n")

# ------------------------------------------------------------------
# 1. LOAD & EDA
# ------------------------------------------------------------------
df = pd.read_csv("creditcard.csv")
log("## 1. Exploratory Data Analysis\n")
log(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
log(f"Missing values: {df.isna().sum().sum()}")

class_counts = df["Class"].value_counts()
fraud_rate = df["Class"].mean() * 100
log(f"\nClass distribution:")
log(f"- Legitimate (0): {class_counts[0]:,} ({100 - fraud_rate:.3f}%)")
log(f"- Fraudulent (1): {class_counts[1]:,} ({fraud_rate:.3f}%)")
log(f"\nImbalance ratio: {class_counts[0] / class_counts[1]:.1f} : 1\n")

log("A model that always predicts 'Legitimate' would score "
    f"{100 - fraud_rate:.2f}% accuracy while catching **zero fraud** -- "
    "this is exactly the 'Illusion of Accuracy' trap the project warns "
    "against. Accuracy is discarded as an evaluation metric for this reason; "
    "Precision, Recall, F1, and ROC-AUC are used instead.\n")

# ------------------------------------------------------------------
# 2. LEAK-FREE TRAIN/TEST SPLIT
# ------------------------------------------------------------------
log("## 2. Train/Test Split (Stratified, Pre-Resampling)\n")

X = df.drop(columns=["Class"]).values
y = df["Class"].values
feature_names = df.drop(columns=["Class"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

log(f"Train set: {X_train.shape[0]:,} rows | Fraud rate: {y_train.mean()*100:.3f}%")
log(f"Test set:  {X_test.shape[0]:,} rows | Fraud rate: {y_test.mean()*100:.3f}%")
log("\nSplit is stratified to preserve the exact 0.172% imbalance in both sets. "
    "**Critically, the split happens BEFORE any SMOTE or scaling is applied** -- "
    "the test set stays untouched and reflects the real-world imbalance, per "
    "the project's 'Golden Rule of Validation.'\n")

# ------------------------------------------------------------------
# 3 & 4. MODEL TRAINING WITH LEAK-FREE SMOTE + GRIDSEARCHCV
# ------------------------------------------------------------------
log("## 3. Model Training (Leak-Free SMOTE + Hyperparameter Tuning)\n")
log("SMOTE is applied **only inside each training fold** during "
    "cross-validation (never to the validation fold or test set), matching "
    "the `imblearn.pipeline.Pipeline` production standard described in the "
    "brief. Because the `imblearn` package could not be installed in this "
    "offline sandbox, the same leak-free guarantee is implemented by hand "
    "via a custom sklearn-compatible wrapper (`SMOTEWrappedClassifier`) whose "
    "`.fit()` resamples internally and whose `.predict()` never resamples. "
    "See `smote_pipeline.py` / `smote_utils.py`.\n")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

results = {}

# ---- Logistic Regression (needs scaling; StandardScaler lives inside the
#      wrapper's .fit() so it too is fit only on the training fold) ----
log("### Logistic Regression\n")
lr_wrapper = SMOTEWrappedClassifier(
    base_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    scaler=StandardScaler(),
    k_neighbors=5,
    random_state=RANDOM_STATE,
)
lr_param_grid = {
    "base_estimator__C": [0.01, 0.1, 1.0, 10.0],
}
lr_grid = GridSearchCV(
    lr_wrapper, lr_param_grid, scoring="roc_auc", cv=cv, n_jobs=4
)
lr_grid.fit(X_train, y_train)
log(f"Best params: {lr_grid.best_params_}")
log(f"Best CV ROC-AUC: {lr_grid.best_score_:.4f}\n")
results["Logistic Regression"] = lr_grid.best_estimator_

# ---- Random Forest (no scaling needed -- tree splits are scale-invariant) ----
log("### Random Forest\n")
rf_wrapper = SMOTEWrappedClassifier(
    base_estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    scaler=None,
    k_neighbors=5,
    random_state=RANDOM_STATE,
)
rf_param_grid = {
    "base_estimator__n_estimators": [100],
    "base_estimator__max_depth": [10, None],
}
rf_grid = GridSearchCV(
    rf_wrapper, rf_param_grid, scoring="roc_auc", cv=cv, n_jobs=2
)
rf_grid.fit(X_train, y_train)
log(f"Best params: {rf_grid.best_params_}")
log(f"Best CV ROC-AUC: {rf_grid.best_score_:.4f}\n")
results["Random Forest"] = rf_grid.best_estimator_

# ------------------------------------------------------------------
# 5. FINAL EVALUATION ON UNTOUCHED TEST SET
# ------------------------------------------------------------------
log("## 4. Final Evaluation on Held-Out Test Set\n")
log("The test set was never touched by SMOTE or scaling fitting -- it "
    "reflects the true 0.172% real-world imbalance, so these numbers are "
    "an honest estimate of production performance.\n")

eval_rows = []
roc_data = {}
for name, model in results.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    eval_rows.append((name, acc, prec, rec, f1, auc))
    roc_data[name] = roc_curve(y_test, y_proba)

    log(f"### {name}\n")
    log(f"- Accuracy (reference only, NOT used to select model): {acc:.4f}")
    log(f"- **Precision**: {prec:.4f}  (of flagged transactions, how many were really fraud)")
    log(f"- **Recall**: {rec:.4f}  (of actual frauds, how many were caught)")
    log(f"- **F1-score**: {f1:.4f}")
    log(f"- **ROC-AUC**: {auc:.4f}")
    log(f"- Confusion Matrix [[TN, FP], [FN, TP]]:\n{cm}\n")

eval_df = pd.DataFrame(eval_rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
log("### Model Comparison Summary\n")
log(eval_df.round(4).to_markdown(index=False))

best_model_name = eval_df.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
log(f"\n**Best model by ROC-AUC: {best_model_name}**\n")
log("Note: Random Forest is generally expected to edge out Logistic Regression "
    "on this kind of problem since fraud patterns are non-linear and tree "
    "ensembles are immune to feature-scale distortion, while Logistic Regression "
    "offers superior interpretability via coefficient inspection if regulatory "
    "explainability is a requirement.\n")

with open("fraud_report.md", "w") as f:
    f.write("\n".join(report_lines))

# save evaluation data for chart generation
import pickle
with open("eval_artifacts.pkl", "wb") as f:
    pickle.dump({
        "eval_df": eval_df,
        "roc_data": roc_data,
        "y_test": y_test,
        "results": {k: None for k in results},  # models not pickled (too large/complex wrapper)
        "confusion_matrices": {name: confusion_matrix(y_test, results[name].predict(X_test)) for name in results},
    }, f)

print("\nPipeline complete. Report written to fraud_report.md")
