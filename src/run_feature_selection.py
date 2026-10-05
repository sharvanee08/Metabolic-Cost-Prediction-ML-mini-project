from load_data import load_dataset
from preprocessing import prepare_subject, get_feature_sets
from feature_selection import forward_stepwise_selection, pca_reduction


if __name__ == "__main__":
    data = load_dataset()

    X = data["subject_1_X"]
    y = data["subject_1_y"]

    X, y = prepare_subject(
        X, y, remove_negative=True
    )

    print("\n=== FORWARD STEPWISE SELECTION ===")

    selected, history = forward_stepwise_selection(
        X, y, max_features=10
    )

    print("\nSelected features:")
    print(selected)

    print("\n=== PCA ===")

    X_pca, pca = pca_reduction(
        X, variance=0.988
    )

    print("Original shape:", X.shape)
    print("PCA shape:", X_pca.shape)