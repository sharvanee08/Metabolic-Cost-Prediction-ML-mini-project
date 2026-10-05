from scipy.io import loadmat


def load_dataset(path="data/processed_data.mat"):
    data = loadmat(path)

    dataset = {
        "subject_1_X": data["eley_data"],
        "subject_1_y": data["eley_metabolics"].ravel(),
        "subject_2_X": data["michael_data"],
        "subject_2_y": data["michael_metabolics"].ravel(),
        "feature_names": data["data_labels"].ravel(),
    }

    return dataset


if __name__ == "__main__":
    dataset = load_dataset()

    print("Subject 1:", dataset["subject_1_X"].shape)
    print("Subject 1 target:", dataset["subject_1_y"].shape)
    print("Subject 2:", dataset["subject_2_X"].shape)
    print("Subject 2 target:", dataset["subject_2_y"].shape)
    print("Features:", len(dataset["feature_names"]))