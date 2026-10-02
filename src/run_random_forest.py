from load_data import load_dataset
from preprocessing import prepare_subject, get_feature_sets
from random_forest import cross_validate_random_forest


def run_subject(name, X, y):
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    X, y = prepare_subject(X, y, remove_negative=True)

    feature_sets = get_feature_sets(X)

    for feature_name, X_features in feature_sets.items():
        print(f"\n--- {feature_name} ---")

        mse = cross_validate_random_forest(
            X_features,
            y,
            n_estimators=200,
            n_splits=5,
            random_state=42,
        )

        print(f"Random Forest CV MSE: {mse:.6f}")


if __name__ == "__main__":
    data = load_dataset()

    run_subject(
        "Subject 1",
        data["subject_1_X"],
        data["subject_1_y"],
    )

    run_subject(
        "Subject 2",
        data["subject_2_X"],
        data["subject_2_y"],
    )