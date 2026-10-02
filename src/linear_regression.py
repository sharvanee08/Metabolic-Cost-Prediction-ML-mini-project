import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def cross_validate_lasso(
    X,
    y,
    alphas=None,
    n_splits=10,
    random_state=42,
):
    """LASSO regression with 10-fold CV."""

    if alphas is None:
        alphas = np.logspace(-4, 1, 30)

    rng = np.random.default_rng(random_state)

    # Randomize samples before CV, matching the experiment.
    permutation = rng.permutation(len(y))
    X = X[permutation]
    y = y[permutation]

    kfold = KFold(
        n_splits=n_splits,
        shuffle=False,
    )

    results = []

    for alpha in alphas:
        fold_mse = []

        for train_idx, test_idx in kfold.split(X):

            model = Lasso(
                alpha=alpha,
                max_iter=10000,
            )

            model.fit(
                X[train_idx],
                y[train_idx],
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

        results.append({
            "alpha": alpha,
            "mse": mean_mse,
        })

    best = min(
        results,
        key=lambda r: r["mse"]
    )

    print(
        f"Best alpha: {best['alpha']:.6f} | "
        f"Mean CV MSE: {best['mse']:.6f}"
    )

    return results, best