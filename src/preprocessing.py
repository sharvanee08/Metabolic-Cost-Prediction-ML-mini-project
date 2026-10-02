import numpy as np


def standardize_features(X):
    """Zero mean, unit variance feature scaling."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)

    return (X - mean) / std


def prepare_subject(X, y, remove_negative=False):
    """Prepare one subject exactly for the ML experiments."""
    if remove_negative:
        mask = y >= 0
        X = X[mask]
        y = y[mask]

    X = standardize_features(X)

    return X, y


def get_feature_sets(X):
    """Return the four feature configurations used in the paper."""
    return {
        "all": X,
        "step_emg": X[:, :25],
        "step_only": X[:, :9],
        "emg_only": X[:, 9:25],
    }