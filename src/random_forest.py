import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def cross_validate_random_forest(
    X,
    y,
    n_estimators=200,
    max_depth=None,
    n_splits=5,
    random_state=42,
):
    """Random Forest regression with 5-fold cross-validation."""

    # Use the same sample ordering/fold structure as the
    # LASSO reproduction.
    rng = np.random.default_rng(random_state)

    permutation = rng.permutation(len(y))
    X = X[permutation]
    y = y[permutation]

    kfold = KFold(
        n_splits=n_splits,
        shuffle=False,
    )

    fold_mse = []

    for train_idx, test_idx in kfold.split(X):

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(
            X[train_idx],
            y[train_idx]
        )

        prediction = model.predict(
            X[test_idx]
        )

        mse = mean_squared_error(
            y[test_idx],
            prediction
        )

        fold_mse.append(mse)

    mean_mse = np.mean(fold_mse)

    return mean_mse