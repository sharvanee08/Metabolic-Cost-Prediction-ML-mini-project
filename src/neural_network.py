import numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class BayesianRegularizedNN:
    """
    One-hidden-layer tanh neural network.

    Approximates MATLAB trainbr:
    - tanh hidden layer
    - linear output
    - L2/Bayesian-style regularization
    - Levenberg-Marquardt optimization
    """

    def __init__(self, n_hidden, max_nfev=100, random_state=42):
        self.n_hidden = n_hidden
        self.max_nfev = max_nfev
        self.random_state = random_state

    def _unpack(self, w, n_features):
        h = self.n_hidden

        n_w1 = n_features * h

        W1 = w[:n_w1].reshape(n_features, h)
        b1 = w[n_w1:n_w1 + h]

        start = n_w1 + h
        W2 = w[start:start + h]
        b2 = w[start + h]

        return W1, b1, W2, b2

    def _forward(self, X, w):
        W1, b1, W2, b2 = self._unpack(w, X.shape[1])

        hidden = np.tanh(X @ W1 + b1)
        output = hidden @ W2 + b2

        return hidden, output

    def _residuals(self, w, X, y, alpha):
        _, output = self._forward(X, w)

        error = output - y
        regularization = np.sqrt(alpha) * w

        return np.concatenate([error, regularization])

    def _jacobian(self, w, X, y, alpha):
        W1, b1, W2, b2 = self._unpack(w, X.shape[1])

        hidden = np.tanh(X @ W1 + b1)
        derivative = 1.0 - hidden ** 2

        n = X.shape[0]
        nf = X.shape[1]
        h = self.n_hidden
        p = len(w)

        J = np.zeros((n + p, p))

        # W1 derivatives
        for j in range(h):
            start = j * nf
            end = start + nf

            J[:n, start:end] = (
                X * (derivative[:, j] * W2[j])[:, None]
            )

        # b1 derivatives
        start = nf * h
        J[:n, start:start + h] = derivative * W2

        # W2 derivatives
        start = nf * h + h
        J[:n, start:start + h] = hidden

        # b2 derivative
        J[:n, start + h] = 1.0

        # regularization derivatives
        J[n:, :] = np.sqrt(alpha) * np.eye(p)

        return J

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)

        nf = X.shape[1]
        h = self.n_hidden

        n_weights = nf * h + h + h + 1

        weights = rng.normal(
            0,
            0.05,
            n_weights
        )

        # Bayesian-style regularization strength
        alpha = 0.001

        result = least_squares(
            self._residuals,
            weights,
            jac=self._jacobian,
            args=(X, y, alpha),
            method="lm",
            max_nfev=self.max_nfev,
        )

        self.weights_ = result.x

        return self

    def predict(self, X):
        _, output = self._forward(X, self.weights_)
        return output


def cross_validate_neurons(
    X,
    y,
    max_neurons=14,
    n_splits=5,
    random_state=42,
):
    """
    Tune 1–14 hidden neurons using 5-fold CV.
    """

    # Same idea as reference: randomize samples before CV.
    rng = np.random.default_rng(random_state)

    permutation = rng.permutation(len(y))

    X = X[permutation]
    y = y[permutation]

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
                max_nfev=100,
                random_state=random_state,
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
            "neurons": n_hidden,
            "mse": mean_mse,
        })

        print(
            f"Neurons: {n_hidden:2d} | "
            f"Mean CV MSE: {mean_mse:.6f}"
        )

    return results