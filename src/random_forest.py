from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np


def cross_validate_random_forest(
    X,
    y,
    n_estimators=200,
    max_depth=None,
    n_splits=5,
    random_state=42,
):
    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_mse = []

    for train_idx, test_idx in kfold.split(X):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(X[train_idx], y[train_idx])

        prediction = model.predict(X[test_idx])

        mse = mean_squared_error(
            y[test_idx],
            prediction,
        )

        fold_mse.append(mse)

    mean_mse = np.mean(fold_mse)

    return mean_mse