import os
import random

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_params():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def collect_images(raw_dir):
    rows = []

    for class_name in sorted(os.listdir(raw_dir)):
        class_dir = os.path.join(raw_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        for filename in sorted(os.listdir(class_dir)):
            file_path = os.path.join(class_dir, filename)

            if os.path.isfile(file_path):
                rows.append(
                    {
                        "path": file_path,
                        "label": class_name,
                    }
                )

    return pd.DataFrame(rows)


def main():
    params = load_params()

    seed = params["seed"]
    raw_dir = params["data"]["raw_dir"]
    processed_dir = params["data"]["processed_dir"]

    set_seed(seed)

    os.makedirs(processed_dir, exist_ok=True)

    df = collect_images(raw_dir)

    train_val, test = train_test_split(
        df,
        test_size=params["split"]["test_size"],
        random_state=seed,
        stratify=df["label"],
    )

    val_ratio = params["split"]["val_size"] / (1 - params["split"]["test_size"])

    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_val["label"],
    )

    train.to_csv(
        os.path.join(processed_dir, "train.csv"),
        index=False,
    )

    val.to_csv(
        os.path.join(processed_dir, "val.csv"),
        index=False,
    )

    test.to_csv(
        os.path.join(processed_dir, "test.csv"),
        index=False,
    )

    print(f"Total images: {len(df)}")
    print(f"Training images: {len(train)}")
    print(f"Validation images: {len(val)}")
    print(f"Test images: {len(test)}")


if __name__ == "__main__":
    main()
