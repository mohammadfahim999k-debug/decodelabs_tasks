"""
Manual SMOTE (Synthetic Minority Over-sampling Technique) implementation.

Used in place of the `imblearn` package, which could not be installed in this
offline sandbox (no network access). This implements the exact algorithm
described in Chawla et al. (2002): for each minority-class sample, find its
k nearest minority-class neighbors and generate synthetic points by linear
interpolation:

    x_new = x_i + lambda * (x_nn - x_i),   lambda ~ Uniform(0, 1)

This must ONLY ever be fit on the training fold (see pipeline.py) to avoid
data leakage into the test/validation set.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_resample(X, y, minority_label=1, k_neighbors=5, random_state=42):
    """
    Oversample the minority class in (X, y) via SMOTE interpolation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    minority_label : the class value to oversample
    k_neighbors : number of nearest neighbors to interpolate between
    random_state : for reproducibility

    Returns
    -------
    X_resampled, y_resampled : arrays with minority class oversampled to
                                match the majority class count (1:1 balance)
    """
    rng = np.random.default_rng(random_state)

    X = np.asarray(X)
    y = np.asarray(y)

    minority_mask = (y == minority_label)
    X_minority = X[minority_mask]
    n_minority = X_minority.shape[0]
    n_majority = (~minority_mask).sum()

    n_to_generate = n_majority - n_minority
    if n_to_generate <= 0:
        return X, y  # already balanced or minority is larger

    k = min(k_neighbors, n_minority - 1)
    if k < 1:
        raise ValueError("Not enough minority samples to run SMOTE.")

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_minority)
    _, neighbor_idx = nn.kneighbors(X_minority)
    neighbor_idx = neighbor_idx[:, 1:]  # drop self-match (first column)

    # Vectorized generation (no per-sample Python loop -- required at this scale)
    sample_idx = rng.integers(0, n_minority, size=n_to_generate)
    neighbor_slot = rng.integers(0, k, size=n_to_generate)
    neighbor_choice = neighbor_idx[sample_idx, neighbor_slot]
    lam = rng.uniform(0, 1, size=(n_to_generate, 1))

    x_i = X_minority[sample_idx]
    x_nn = X_minority[neighbor_choice]
    synthetic_samples = x_i + lam * (x_nn - x_i)

    X_resampled = np.vstack([X, synthetic_samples])
    y_resampled = np.concatenate([y, np.full(n_to_generate, minority_label)])

    # shuffle so the classifier doesn't see all synthetic points at the tail
    shuffle_idx = rng.permutation(len(y_resampled))
    return X_resampled[shuffle_idx], y_resampled[shuffle_idx]
