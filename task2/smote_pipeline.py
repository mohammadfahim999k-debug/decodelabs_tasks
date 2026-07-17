"""
Leak-free SMOTE-wrapped classifier, sklearn-compatible.

Standard sklearn.pipeline.Pipeline only supports transformers with a
`transform(X)` method that cannot touch `y` and cannot change the number of
rows. SMOTE needs to modify both X and y and change row counts -- this is
exactly the reason the `imblearn` package exists (its Pipeline calls
`fit_resample` instead of `fit_transform`). Since imblearn is unavailable
offline here, this class reproduces the same leak-free guarantee by hand:

  - `.fit(X, y)`  -> applies SMOTE to (X, y) internally, THEN fits the
                     wrapped classifier on the resampled data.
  - `.predict(X)` / `.predict_proba(X)` -> pass straight through to the
                     wrped classifier, NO resampling applied.

Because SMOTE only ever runs inside `.fit()`, and GridSearchCV / cross-
validation call `.fit()` separately on each training fold, this guarantees
SMOTE never sees the held-out validation or test fold -- satisfying the
"Golden Rule of Validation" from the project brief.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from smote_utils import smote_resample


class SMOTEWrappedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator=None, k_neighbors=5, random_state=42,
                 scaler=None, majority_cap=20000):
        self.base_estimator = base_estimator
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.scaler = scaler
        # Hybrid resampling: cap the majority class via random undersampling
        # BEFORE running SMOTE on the minority class -- a standard technique
        # that keeps training sets computationally tractable at scale while
        # still exposing the classifier to a realistic imbalanced ratio.
        # Only ever applied inside .fit(), so it never touches the held-out
        # test/validation fold and introduces no leakage.
        self.majority_cap = majority_cap

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)

        majority_idx = np.where(y == 0)[0]
        minority_idx = np.where(y == 1)[0]
        if len(majority_idx) > self.majority_cap:
            majority_idx = rng.choice(majority_idx, size=self.majority_cap, replace=False)
        keep_idx = np.concatenate([majority_idx, minority_idx])
        X_capped, y_capped = X[keep_idx], y[keep_idx]

        X_fit = X_capped
        if self.scaler is not None:
            self.scaler_ = clone(self.scaler)
            X_fit = self.scaler_.fit_transform(X_fit)

        X_res, y_res = smote_resample(
            X_fit, y_capped, minority_label=1,
            k_neighbors=self.k_neighbors, random_state=self.random_state
        )
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X_res, y_res)
        self.classes_ = self.estimator_.classes_
        return self

    def _transform_for_predict(self, X):
        if self.scaler is not None:
            return self.scaler_.transform(X)
        return X

    def predict(self, X):
        return self.estimator_.predict(self._transform_for_predict(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self._transform_for_predict(X))
