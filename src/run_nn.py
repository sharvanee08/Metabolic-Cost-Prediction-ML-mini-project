from load_data import load_dataset
from preprocessing import prepare_subject, get_feature_sets
from neural_network import cross_validate_neurons


def run_subject(name, X, y, remove_negative=False):
    X, y = prepare_subject(X, y, remove_negative)
    feature_sets = get_feature_sets(X)

    print(f"\n{'=' * 50}")
    print(f"{name}")
    print(f"Samples: {len(y)}")
    print(f"{'=' * 50}")

    for feature_name, X_features in feature_sets.items():
        print(f"\n--- {feature_name} ---")

        results = cross_validate_neurons(
            X_features,
            y,
            max_neurons=14,
            n_splits=5,
            random_state=42,
        )

        best = min(results, key=lambda r: r["mse"])

        print(
            f"BEST → {best['neurons']} neurons | "
            f"MSE = {best['mse']:.6f}"
        )


if __name__ == "__main__":
    data = load_dataset()

    run_subject(
        "Subject 1",
        data["subject_1_X"],
        data["subject_1_y"],
        remove_negative=True,
    )

    run_subject(
        "Subject 2",
        data["subject_2_X"],
        data["subject_2_y"],
        remove_negative=False,
    )