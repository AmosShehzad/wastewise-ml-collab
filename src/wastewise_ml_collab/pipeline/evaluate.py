import json
import os
import subprocess

import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class GarbageDataset(Dataset):
    def __init__(self, dataframe, class_to_idx, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = Image.open(row["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row["label"]]

        return image, label


def get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main():
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    test_csv = os.path.join(
        params["data"]["processed_dir"],
        "test.csv",
    )

    test_df = pd.read_csv(test_csv)

    checkpoint = torch.load(
        "models/model.pth",
        map_location="cpu",
        weights_only=False,
    )

    classes = checkpoint["classes"]

    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}

    image_size = params["train"]["image_size"]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = GarbageDataset(
        test_df,
        class_to_idx,
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=params["train"]["batch_size"],
        shuffle=False,
        num_workers=params["train"]["num_workers"],
    )

    model = models.mobilenet_v2(weights=None)

    num_features = model.classifier[1].in_features

    model.classifier[1] = torch.nn.Linear(
        num_features,
        len(classes),
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
    )

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "git_sha": get_git_sha(),
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Git SHA: {metrics['git_sha']}")
    print("Metrics saved to metrics.json")


if __name__ == "__main__":
    main()
