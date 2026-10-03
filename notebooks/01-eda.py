# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: wastewise-ml-collab (3.12.0)
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.02034, "end_time": "2024-07-18T16:22:40.410778", "exception": false, "start_time": "2024-07-18T16:22:40.390438", "status": "completed"}
# # Garbage Classification using PyTorch
#
# Garbage segregation involves separating wastes according to how it's handled or processed. It's important for recycling as some materials are recyclable and others are not.
#
#
# ![Garbage Bins](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwebstockreview.net%2Fimages%2Fgarbage-clipart-wastebin-16.png&f=1&nofb=1)
#
#
# In this notebook we'll use PyTorch for classifying trash into various categories like metal, cardboard, etc.

# %% [markdown] papermill={"duration": 0.019289, "end_time": "2024-07-18T16:22:40.448789", "exception": false, "start_time": "2024-07-18T16:22:40.4295", "status": "completed"}
# Let us start by importing the libraries:

# %% papermill={"duration": 1.989059, "end_time": "2024-07-18T16:22:42.455503", "exception": false, "start_time": "2024-07-18T16:22:40.466444", "status": "completed"}
# Import necessary libraries
import torch  # PyTorch library for deep learning
from torch.utils.data import (
    random_split,
)  # For splitting datasets into train and validation sets
import torchvision.models as models  # Pre-trained models available in torchvision
import torch.nn as nn  # Neural network module in PyTorch
import torch.nn.functional as F  # Functional interface for operations on tensors

import matplotlib.pyplot as plt  # Matplotlib for plotting

# # %matplotlib inline  # Magic command to display plots inline in Jupyter Notebook or IPython

# %% [markdown] papermill={"duration": 0.017584, "end_time": "2024-07-18T16:22:42.49127", "exception": false, "start_time": "2024-07-18T16:22:42.473686", "status": "completed"}
# Let us see the classes present in the dataset:

# %% papermill={"duration": 1.27334, "end_time": "2024-07-18T16:22:43.783098", "exception": false, "start_time": "2024-07-18T16:22:42.509758", "status": "completed"}
data_dir = "../data/raw/garbage-dataset"
from wastewise_ml_collab.data_utils import count_images_by_class

class_count = count_images_by_class(data_dir)

# Plot the number of images in each class
plt.figure(figsize=(10, 6))
plt.bar(class_count.keys(), class_count.values(), color="skyblue")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.title("Number of Images in Each Class")
plt.xticks(rotation=45, ha="right")
plt.show()

# %% [markdown] papermill={"duration": 0.018424, "end_time": "2024-07-18T16:22:43.82138", "exception": false, "start_time": "2024-07-18T16:22:43.802956", "status": "completed"}
# ## Transformations:

# %% [markdown] papermill={"duration": 0.018949, "end_time": "2024-07-18T16:22:43.859124", "exception": false, "start_time": "2024-07-18T16:22:43.840175", "status": "completed"}
# Now, let's apply transformations to the dataset and import it for use.

# %% papermill={"duration": 6.474546, "end_time": "2024-07-18T16:22:50.352846", "exception": false, "start_time": "2024-07-18T16:22:43.8783", "status": "completed"}
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transformations = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()]
)

dataset = ImageFolder(data_dir, transform=transformations)


# %% [markdown] papermill={"duration": 0.017564, "end_time": "2024-07-18T16:22:50.388929", "exception": false, "start_time": "2024-07-18T16:22:50.371365", "status": "completed"}
# Let's create a helper function to see the image and its corresponding label:


# %% papermill={"duration": 0.028813, "end_time": "2024-07-18T16:22:50.436356", "exception": false, "start_time": "2024-07-18T16:22:50.407543", "status": "completed"}
def show_sample(img, label):
    print("Label:", dataset.classes[label], "(Class No: " + str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))


# %% papermill={"duration": 0.223735, "end_time": "2024-07-18T16:22:50.678528", "exception": false, "start_time": "2024-07-18T16:22:50.454793", "status": "completed"}
img, label = dataset[12]
show_sample(img, label)

# %% [markdown] papermill={"duration": 0.019293, "end_time": "2024-07-18T16:22:50.717403", "exception": false, "start_time": "2024-07-18T16:22:50.69811", "status": "completed"}
# # Loading and Splitting Data:

# %% papermill={"duration": 0.033214, "end_time": "2024-07-18T16:22:50.769362", "exception": false, "start_time": "2024-07-18T16:22:50.736148", "status": "completed"}
random_seed = 42
torch.manual_seed(random_seed)

# %% [markdown] papermill={"duration": 0.01908, "end_time": "2024-07-18T16:22:50.807958", "exception": false, "start_time": "2024-07-18T16:22:50.788878", "status": "completed"}
# We'll split the dataset into training, validation and test sets:

# %% papermill={"duration": 0.030825, "end_time": "2024-07-18T16:22:50.857776", "exception": false, "start_time": "2024-07-18T16:22:50.826951", "status": "completed"}
len(dataset)
#  8495 ,

# %% papermill={"duration": 0.032452, "end_time": "2024-07-18T16:22:50.91006", "exception": false, "start_time": "2024-07-18T16:22:50.877608", "status": "completed"}
train_ds, test_ds = random_split(
    dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
)
train_ds, val_ds = random_split(
    train_ds, [int(0.8 * len(train_ds)), len(train_ds) - int(0.8 * len(train_ds))]
)

# %% papermill={"duration": 0.029783, "end_time": "2024-07-18T16:22:50.959202", "exception": false, "start_time": "2024-07-18T16:22:50.929419", "status": "completed"}
print(len(train_ds), len(test_ds), len(val_ds))

# %% papermill={"duration": 0.029296, "end_time": "2024-07-18T16:22:51.009034", "exception": false, "start_time": "2024-07-18T16:22:50.979738", "status": "completed"}
from torch.utils.data.dataloader import DataLoader

batch_size = 32

# %% [markdown] papermill={"duration": 0.020214, "end_time": "2024-07-18T16:22:51.049714", "exception": false, "start_time": "2024-07-18T16:22:51.0295", "status": "completed"}
# Now, we'll create training and validation dataloaders using `DataLoader`.

# %% papermill={"duration": 0.030204, "end_time": "2024-07-18T16:22:51.100551", "exception": false, "start_time": "2024-07-18T16:22:51.070347", "status": "completed"}
train_dl = DataLoader(
    train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)

# %% [markdown] papermill={"duration": 0.019125, "end_time": "2024-07-18T16:22:51.139244", "exception": false, "start_time": "2024-07-18T16:22:51.120119", "status": "completed"}
# This is a helper function to visualize batches:

# %% papermill={"duration": 0.031343, "end_time": "2024-07-18T16:22:51.190006", "exception": false, "start_time": "2024-07-18T16:22:51.158663", "status": "completed"}
from torchvision.utils import make_grid


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break


# %% papermill={"duration": 5.269658, "end_time": "2024-07-18T16:22:56.47911", "exception": false, "start_time": "2024-07-18T16:22:51.209452", "status": "completed"}
show_batch(train_dl)


# %% [markdown] papermill={"duration": 0.022876, "end_time": "2024-07-18T16:22:56.52534", "exception": false, "start_time": "2024-07-18T16:22:56.502464", "status": "completed"}
# # Model Base:

# %% [markdown] papermill={"duration": 0.022101, "end_time": "2024-07-18T16:22:56.569486", "exception": false, "start_time": "2024-07-18T16:22:56.547385", "status": "completed"}
# Let's create the model base:


# %% papermill={"duration": 0.041071, "end_time": "2024-07-18T16:22:56.633141", "exception": false, "start_time": "2024-07-18T16:22:56.59207", "status": "completed"}
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch + 1, result["train_loss"], result["val_loss"], result["val_acc"]
            )
        )


# %% [markdown] papermill={"duration": 0.021167, "end_time": "2024-07-18T16:22:56.676466", "exception": false, "start_time": "2024-07-18T16:22:56.655299", "status": "completed"}
# We'll be using ResNet50 for classifying images:


# %% papermill={"duration": 1.631212, "end_time": "2024-07-18T16:22:58.329484", "exception": false, "start_time": "2024-07-18T16:22:56.698272", "status": "completed"}
class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


model = ResNet()


# %% [markdown] papermill={"duration": 0.022561, "end_time": "2024-07-18T16:22:58.375825", "exception": false, "start_time": "2024-07-18T16:22:58.353264", "status": "completed"}
# ## Porting to GPU:

# %% [markdown] papermill={"duration": 0.022962, "end_time": "2024-07-18T16:22:58.421648", "exception": false, "start_time": "2024-07-18T16:22:58.398686", "status": "completed"}
# GPUs tend to perform faster calculations than CPU. Let's take this advantage and use GPU for computation:


# %% papermill={"duration": 0.037058, "end_time": "2024-07-18T16:22:58.481011", "exception": false, "start_time": "2024-07-18T16:22:58.443953", "status": "completed"}
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# %% papermill={"duration": 0.0328, "end_time": "2024-07-18T16:22:58.536514", "exception": false, "start_time": "2024-07-18T16:22:58.503714", "status": "completed"}
device = get_default_device()
device

# %% papermill={"duration": 0.073775, "end_time": "2024-07-18T16:22:58.632733", "exception": false, "start_time": "2024-07-18T16:22:58.558958", "status": "completed"}
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)


# %% [markdown] papermill={"duration": 0.022552, "end_time": "2024-07-18T16:22:58.677948", "exception": false, "start_time": "2024-07-18T16:22:58.655396", "status": "completed"}
# # Training the Model:

# %% [markdown] papermill={"duration": 0.022431, "end_time": "2024-07-18T16:22:58.722974", "exception": false, "start_time": "2024-07-18T16:22:58.700543", "status": "completed"}
# This is the function for fitting the model.


# %% papermill={"duration": 0.037707, "end_time": "2024-07-18T16:22:58.783527", "exception": false, "start_time": "2024-07-18T16:22:58.74582", "status": "completed"}
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# %% papermill={"duration": 0.678151, "end_time": "2024-07-18T16:22:59.484153", "exception": false, "start_time": "2024-07-18T16:22:58.806002", "status": "completed"}
model = to_device(ResNet(), device)

# %% papermill={"duration": 24.219967, "end_time": "2024-07-18T16:23:23.726721", "exception": false, "start_time": "2024-07-18T16:22:59.506754", "status": "completed"}
evaluate(model, val_dl)

# %% [markdown] papermill={"duration": 0.022994, "end_time": "2024-07-18T16:23:23.772605", "exception": false, "start_time": "2024-07-18T16:23:23.749611", "status": "completed"}
# Let's start training the model:

# %% papermill={"duration": 1530.952524, "end_time": "2024-07-18T16:48:54.749464", "exception": false, "start_time": "2024-07-18T16:23:23.79694", "status": "completed"}
num_epochs = 8
opt_func = torch.optim.Adam
lr = 5.5e-5

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


# %% papermill={"duration": 0.289863, "end_time": "2024-07-18T16:48:55.063771", "exception": false, "start_time": "2024-07-18T16:48:54.773908", "status": "completed"}
def plot_accuracies(history):
    accuracies = [x["val_acc"] for x in history]
    plt.plot(accuracies, "-x")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs. No. of epochs")


plot_accuracies(history)


# %% papermill={"duration": 0.328157, "end_time": "2024-07-18T16:48:55.41707", "exception": false, "start_time": "2024-07-18T16:48:55.088913", "status": "completed"}
def plot_losses(history):
    train_losses = [x.get("train_loss") for x in history]
    val_losses = [x["val_loss"] for x in history]
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss vs. No. of epochs")


plot_losses(history)


# %% [markdown] papermill={"duration": 0.024847, "end_time": "2024-07-18T16:48:55.466973", "exception": false, "start_time": "2024-07-18T16:48:55.442126", "status": "completed"}
# # Visualizing Predictions:


# %% papermill={"duration": 0.036619, "end_time": "2024-07-18T16:48:55.52892", "exception": false, "start_time": "2024-07-18T16:48:55.492301", "status": "completed"}
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]


# %% [markdown] papermill={"duration": 0.024787, "end_time": "2024-07-18T16:48:55.579355", "exception": false, "start_time": "2024-07-18T16:48:55.554568", "status": "completed"}
# Let us see the model's predictions on the test dataset:

# %% papermill={"duration": 0.328298, "end_time": "2024-07-18T16:48:55.932925", "exception": false, "start_time": "2024-07-18T16:48:55.604627", "status": "completed"}
img, label = test_ds[17]
plt.imshow(img.permute(1, 2, 0))
print("Label:", dataset.classes[label], ", Predicted:", predict_image(img, model))

# %% papermill={"duration": 0.27354, "end_time": "2024-07-18T16:48:56.232829", "exception": false, "start_time": "2024-07-18T16:48:55.959289", "status": "completed"}
img, label = test_ds[23]
plt.imshow(img.permute(1, 2, 0))
print("Label:", dataset.classes[label], ", Predicted:", predict_image(img, model))

# %% papermill={"duration": 0.33627, "end_time": "2024-07-18T16:48:56.597018", "exception": false, "start_time": "2024-07-18T16:48:56.260748", "status": "completed"}
img, label = test_ds[51]
plt.imshow(img.permute(1, 2, 0))
print("Label:", dataset.classes[label], ", Predicted:", predict_image(img, model))

# %% [markdown] papermill={"duration": 0.029011, "end_time": "2024-07-18T16:48:56.656558", "exception": false, "start_time": "2024-07-18T16:48:56.627547", "status": "completed"}
# # Predicting External Images:

# %% [markdown] papermill={"duration": 0.029427, "end_time": "2024-07-18T16:48:56.715693", "exception": false, "start_time": "2024-07-18T16:48:56.686266", "status": "completed"}
# Let's now test with external images.
#
# I'll use `urllib` for downloading external images.

# %% papermill={"duration": 1.653918, "end_time": "2024-07-18T16:48:58.398844", "exception": false, "start_time": "2024-07-18T16:48:56.744926", "status": "completed"}
import urllib.request

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/sumn2u/ml_rest_api/master/test-images/plastic.jpeg",
    "plastic.jpg",
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/sumn2u/ml_rest_api/master/test-images/carboard.jpeg",
    "cardboard.jpg",
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/sumn2u/ml_rest_api/master/test-images/cans.jpeg",
    "cans.jpg",
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/sumn2u/ml_rest_api/master/test-images/wine.jpeg",
    "wine-trash.jpg",
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/sumn2u/ml_rest_api/master/test-images/paper.jpeg",
    "paper-trash.jpg",
)

# %% [markdown] papermill={"duration": 0.029255, "end_time": "2024-07-18T16:48:58.458212", "exception": false, "start_time": "2024-07-18T16:48:58.428957", "status": "completed"}
# Let us load the model. You can load an external pre-trained model too!

# %% papermill={"duration": 0.038864, "end_time": "2024-07-18T16:48:58.526779", "exception": false, "start_time": "2024-07-18T16:48:58.487915", "status": "completed"}
loaded_model = model

# %% [markdown] papermill={"duration": 0.029423, "end_time": "2024-07-18T16:48:58.586449", "exception": false, "start_time": "2024-07-18T16:48:58.557026", "status": "completed"}
# This function takes the image's name and prints the predictions:

# %% papermill={"duration": 0.040574, "end_time": "2024-07-18T16:48:58.656685", "exception": false, "start_time": "2024-07-18T16:48:58.616111", "status": "completed"}
from PIL import Image
from pathlib import Path


def predict_external_image(image_name):
    image = Image.open(Path("./" + image_name))

    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    print("The image resembles", predict_image(example_image, loaded_model) + ".")


# %% papermill={"duration": 0.783229, "end_time": "2024-07-18T16:48:59.469561", "exception": false, "start_time": "2024-07-18T16:48:58.686332", "status": "completed"}
predict_external_image("cans.jpg")

# %% papermill={"duration": 1.327125, "end_time": "2024-07-18T16:49:00.828408", "exception": false, "start_time": "2024-07-18T16:48:59.501283", "status": "completed"}
predict_external_image("wine-trash.jpg")

# %% papermill={"duration": 0.561007, "end_time": "2024-07-18T16:49:01.424537", "exception": false, "start_time": "2024-07-18T16:49:00.86353", "status": "completed"}
predict_external_image("paper-trash.jpg")

# %% papermill={"duration": 0.261473, "end_time": "2024-07-18T16:49:01.722255", "exception": false, "start_time": "2024-07-18T16:49:01.460782", "status": "completed"}
torch.save(model, "/kaggle/working/final_model.pt")

# %% [markdown] papermill={"duration": 0.035608, "end_time": "2024-07-18T16:49:01.794167", "exception": false, "start_time": "2024-07-18T16:49:01.758559", "status": "completed"}
# # Mobile Usage:
# Save model for mobile usage.

# %% papermill={"duration": 0.045867, "end_time": "2024-07-18T16:49:01.875908", "exception": false, "start_time": "2024-07-18T16:49:01.830041", "status": "completed"}
# script = model.to_torchscript()
# torch.jit.save(script,'/kaggle/working/final_mobile_model.t7')
# compiled_model = torch.jit.script(model)
# torch.jit.save(compiled_model, '/kaggle/working/final_mobile_model.pt')

# %% papermill={"duration": 0.044464, "end_time": "2024-07-18T16:49:01.955883", "exception": false, "start_time": "2024-07-18T16:49:01.911419", "status": "completed"}
# model.eval()
# example = torch.rand(1, 3, 224, 224)
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("/kaggle/working/traced_resnet_model.pt")

# %% papermill={"duration": 0.048281, "end_time": "2024-07-18T16:49:02.039013", "exception": false, "start_time": "2024-07-18T16:49:01.990732", "status": "completed"}
example_input = torch.rand(1, 3, 224, 224, dtype=torch.float)
example_input = example_input.to(device)
# model = torch.jit.trace(model, example_input, check_trace=True, check_tolerance=1e-05, optimize=False,)

# %% papermill={"duration": 1.451904, "end_time": "2024-07-18T16:49:03.526721", "exception": false, "start_time": "2024-07-18T16:49:02.074817", "status": "completed"}
model = torch.jit.trace(model, example_input, check_trace=True, check_tolerance=1e-05)

# %% papermill={"duration": 0.23556, "end_time": "2024-07-18T16:49:03.79862", "exception": false, "start_time": "2024-07-18T16:49:03.56306", "status": "completed"}
torch.jit.save(model, "/kaggle/working/traced_resnet_model.pt")

# %% papermill={"duration": 0.249616, "end_time": "2024-07-18T16:49:04.087861", "exception": false, "start_time": "2024-07-18T16:49:03.838245", "status": "completed"}
loaded_compiled_model = torch.jit.load("/kaggle/working/traced_resnet_model.pt")
loaded_compiled_model.eval()

# %% [markdown] papermill={"duration": 0.035112, "end_time": "2024-07-18T16:49:04.158869", "exception": false, "start_time": "2024-07-18T16:49:04.123757", "status": "completed"}
# # Conclusion:
#
# Our model is able to classify garbage with **≈97% accuracy**!
#
# It's great to see the model's predictions on the test set. It works pretty good on external images too!
#
# You can try experimenting with more images and see the results!

# %% [markdown] papermill={"duration": 0.034753, "end_time": "2024-07-18T16:49:04.229334", "exception": false, "start_time": "2024-07-18T16:49:04.194581", "status": "completed"}
# ### If you liked the kernel, don't forget to show some appreciation :)
