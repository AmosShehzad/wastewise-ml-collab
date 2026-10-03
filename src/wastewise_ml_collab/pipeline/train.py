import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def main():
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    seed = params["seed"]
    set_seed(seed)

    train_csv = os.path.join(
        params["data"]["processed_dir"],
        "train.csv",
    )

    train_df = pd.read_csv(train_csv)

    classes = sorted(train_df["label"].unique())

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
        train_df,
        class_to_idx,
        transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=params["train"]["batch_size"],
        shuffle=True,
        num_workers=params["train"]["num_workers"],
        generator=generator,
    )

    weights = models.MobileNet_V2_Weights.DEFAULT

    model = models.mobilenet_v2(weights=weights)

    # Freeze the pretrained feature extractor.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        len(classes),
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.classifier[1].parameters(),
        lr=params["train"]["learning_rate"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    print(f"Using device: {device}")
    print(f"Training images: {len(dataset)}")
    print(f"Classes: {classes}")

    epochs = params["train"]["epochs"]

    for epoch in range(epochs):
        model.train()

        # Keep the frozen feature extractor in evaluation mode.
        model.features.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {running_loss / len(loader):.4f} "
            f"Accuracy: {accuracy:.4f}"
        )

    os.makedirs("models", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
        },
        "models/model.pth",
    )

    print("Model saved to models/model.pth")


if __name__ == "__main__":
    main()
