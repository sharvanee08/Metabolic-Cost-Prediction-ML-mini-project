import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def cross_validate_neurons(
    X,
    y,
    max_neurons=14,
    n_splits=5,
    random_state=42,
):
    """
    Python reproduction of the reference NN experiment.

    Reference:
    - one hidden layer
    - tanh activation
    - linear output
    - 5-fold CV
    - 1–14 hidden neurons
    - Bayesian regularization

    Python approximation:
    MLPRegressor with L2 regularization.
    """

    # Match the reference experiment's random permutation.
    rng = np.random.default_rng(random_state)
    permutation = rng.permutation(len(y))

    X = X[permutation]
    y = y[permutation]

    kfold = KFold(
        n_splits=n_splits,
        shuffle=False,
    )

    results = []

    for n_hidden in range(1, max_neurons + 1):

        fold_mse = []

        for train_idx, test_idx in kfold.split(X):

            model = MLPRegressor(
                hidden_layer_sizes=(n_hidden,),
                activation="tanh",
                solver="lbfgs",
                alpha=1.0,
                max_iter=500,
                random_state=random_state,
            )

            model.fit(
                X[train_idx],
                y[train_idx],
            )

            prediction = model.predict(X[test_idx])

            mse = mean_squared_error(
                y[test_idx],
                prediction
            )

            fold_mse.append(mse)

        mean_mse = np.mean(fold_mse)

        results.append({
            "neurons": n_hidden,
            "mse": mean_mse,
        })

        print(
            f"Neurons: {n_hidden:2d} | "
            f"Mean CV MSE: {mean_mse:.6f}"
        )

    return results