"""Garbage classification training script (ResNet50).

Refactored from the Kaggle notebook:
https://www.kaggle.com/code/sumn2u/garbage-classification-resnet

Run from the repo root:
    uv run python src/train.py --data-dir data/raw/<folder-with-class-folders>
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]  # repo root, so no absolute paths


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet50 garbage classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="folder that contains one sub-folder per class",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "models" / "final_model.pt"
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5.5e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        return F.cross_entropy(out, labels)

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        epoch_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        epoch_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch + 1, result["train_loss"], result["val_loss"], result["val_acc"]
            )
        )


class ResNet(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


def get_default_device():
    """Pick GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move each batch to a device."""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    transformations = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    dataset = ImageFolder(args.data_dir, transform=transformations)
    print(f"Found {len(dataset)} images in {len(dataset.classes)} classes")

    n_train_all = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [n_train_all, len(dataset) - n_train_all])
    n_train = int(0.8 * len(train_ds))
    train_ds, val_ds = random_split(train_ds, [n_train, len(train_ds) - n_train])
    print(len(train_ds), len(test_ds), len(val_ds))

    device = get_default_device()
    train_dl = DataLoader(
        train_ds,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, args.batch_size * 2, num_workers=args.num_workers, pin_memory=True
    )
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    model = to_device(ResNet(len(dataset.classes)), device)
    fit(args.epochs, args.lr, model, train_dl, val_dl)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
