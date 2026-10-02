import numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class BayesianRegularizedNN:
    """
    One-hidden-layer neural network:
    tanh hidden activation + linear output.

    Uses L2/Bayesian-style regularization and numerical
    optimization to approximate MATLAB trainbr behavior.
    """

    def __init__(self, n_hidden, max_iter=500, random_state=42):
        self.n_hidden = n_hidden
        self.max_iter = max_iter
        self.random_state = random_state

    def _unpack(self, w, n_features):
        h = self.n_hidden

        W1 = w[:n_features * h].reshape(n_features, h)
        b1 = w[n_features * h:n_features * h + h]

        start = n_features * h + h
        W2 = w[start:start + h]
        b2 = w[start + h]

        return W1, b1, W2, b2

    def _predict_weights(self, X, w):
        W1, b1, W2, b2 = self._unpack(w, X.shape[1])

        hidden = np.tanh(X @ W1 + b1)
        return hidden @ W2 + b2

    def _residuals(self, w, X, y, alpha, beta):
        prediction_error = np.sqrt(beta) * (
            self._predict_weights(X, w) - y
        )

        regularization = np.sqrt(alpha) * w

        return np.concatenate([prediction_error, regularization])

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)

        n_features = X.shape[1]
        n_weights = n_features * self.n_hidden + self.n_hidden
        n_weights += self.n_hidden + 1

        weights = rng.normal(0, 0.05, n_weights)

        alpha = 1e-3
        beta = 1.0

        for _ in range(10):
            result = least_squares(
                self._residuals,
                weights,
                args=(X, y, alpha, beta),
                max_nfev=self.max_iter,
            )

            weights = result.x

            prediction = self._predict_weights(X, weights)

            data_error = np.sum((prediction - y) ** 2)
            weight_error = np.sum(weights ** 2)

            if weight_error > 0:
                alpha = max(1e-8, len(weights) / (2 * weight_error))

            if data_error > 0:
                beta = max(1e-8, len(y) / (2 * data_error))

        self.weights_ = weights
        return self

    def predict(self, X):
        return self._predict_weights(X, self.weights_)


def cross_validate_neurons(X, y, max_neurons=14, n_splits=5, random_state=42):
    """
    Reproduce the reference experiment:
    test 1–14 hidden neurons using 5-fold CV.
    """

    kfold = KFold(
        n_splits=n_splits,
        shuffle=False
    )

    results = []

    for n_hidden in range(1, max_neurons + 1):
        fold_mse = []

        for train_idx, test_idx in kfold.split(X):
            model = BayesianRegularizedNN(
                n_hidden=n_hidden,
                max_iter=500,
                random_state=random_state
            )

            model.fit(X[train_idx], y[train_idx])

            prediction = model.predict(X[test_idx])

            mse = mean_squared_error(
                y[test_idx],
                prediction
            )

            fold_mse.append(mse)

        mean_mse = np.mean(fold_mse)

        results.append({
            "neurons": n_hidden,
            "mse": mean_mse
        })

        print(
            f"Neurons: {n_hidden:2d} | "
            f"Mean CV MSE: {mean_mse:.6f}"
        )

    return results