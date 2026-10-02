import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def forward_stepwise_selection(X, y, max_features=None):
    """
    Forward step-wise feature selection.

    At each step, add the feature that gives the lowest
    5-fold cross-validation MSE.
    """

    n_features = X.shape[1]

    if max_features is None:
        max_features = n_features

    selected = []
    remaining = list(range(n_features))
    history = []

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for _ in range(max_features):

        best_feature = None
        best_mse = np.inf

        for feature in remaining:

            features = selected + [feature]
            fold_mse = []

            for train_idx, test_idx in kfold.split(X):

                model = LinearRegression()

                model.fit(
                    X[train_idx][:, features],
                    y[train_idx]
                )

                prediction = model.predict(
                    X[test_idx][:, features]
                )

                fold_mse.append(
                    mean_squared_error(
                        y[test_idx],
                        prediction
                    )
                )

            mse = np.mean(fold_mse)

            if mse < best_mse:
                best_mse = mse
                best_feature = feature

        selected.append(best_feature)
        remaining.remove(best_feature)

        history.append({
            "feature": best_feature,
            "mse": best_mse
        })

        print(
            f"Step {len(selected):2d} | "
            f"Feature {best_feature + 1:2d} | "
            f"MSE = {best_mse:.6f}"
        )

    return selected, history


def pca_reduction(X, variance=0.988):
    """
    PCA dimensionality reduction.

    Keeps enough components to explain the requested
    fraction of variance.
    """

    pca = PCA(
        n_components=variance
    )

    X_reduced = pca.fit_transform(X)

    print(
        f"PCA components: {X_reduced.shape[1]}"
    )

    print(
        f"Explained variance: "
        f"{np.sum(pca.explained_variance_ratio_):.4f}"
    )

    return X_reduced, pca