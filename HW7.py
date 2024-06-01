import os
import sys
import gzip
import struct
from typing import List

import numpy as np
from sympy import evaluate
from torch.testing._internal.common_quantization import train_one_epoch
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


def read_images_archive(fi):
    magic, n, rows, columns = struct.unpack(">IIII", fi.read(16))
    assert magic == 0x00000803
    assert rows == 28
    assert columns == 28
    rawbuffer = fi.read()
    assert len(rawbuffer) == n * rows * columns
    rawdata = np.frombuffer(rawbuffer, dtype='>u1', count=n * rows * columns)
    return rawdata.reshape(n, rows, columns).astype(np.float32) / 255.0


def read_labels_archive(fi):
    magic, n = struct.unpack(">II", fi.read(8))
    assert magic == 0x00000801
    rawbuffer = fi.read()
    assert len(rawbuffer) == n
    return np.frombuffer(rawbuffer, dtype='>u1', count=n)


# %%
with gzip.open("train-images-idx3-ubyte.gz", "rb") as fp:
    train_images = read_images_archive(fp)
    print(f"Loaded training images with shape {train_images.shape}!")

with gzip.open("train-labels-idx1-ubyte.gz", "rb") as fp:
    train_labels = read_labels_archive(fp)
    print(f"Loaded training labels with shape {train_labels.shape}!")

with gzip.open("t10k-images-idx3-ubyte.gz", "rb") as fp:
    test_images = read_images_archive(fp)
    print(f"Loaded test images with shape {test_images.shape}!")

with gzip.open("t10k-labels-idx1-ubyte.gz", "rb") as fp:
    test_labels = read_labels_archive(fp)
    print(f"Loaded test labels with shape {test_labels.shape}!")
# %%
# NOTE: visualize training dataset
_nrows = 3
fig, axes = plt.subplots(nrows=_nrows, ncols=10, figsize=(20, 5))
for lbl in range(10):
    for r in range(_nrows):
        ax = axes[r][lbl]
        ax.imshow(train_images[train_labels == lbl][r], cmap='gray')
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(f"{lbl}")
plt.show()


# %%
class MNISTDataset(Dataset):
    def __init__(self, images, labels) -> None:
        if len(images) != len(labels):
            raise ValueError(f"Different number of images ({len(images)}) and labels ({len(labels)})!")

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]  # [0 .. 1]

        image = image.flatten()

        label = self.labels[index]

        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


# %%
class MLP(nn.Module):
    def __init__(self, nin: int, nouts: List[int]) -> None:
        super().__init__()

        sizes = [nin] + nouts

        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(nouts))])

        self.n_layers = len(self.layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)

            def read_images_archive(fi):
                magic, n, rows, columns = struct.unpack(">IIII", fi.read(16))
                assert magic == 0x00000803
                assert rows == 28
                assert columns == 28
                rawbuffer = fi.read()
                assert len(rawbuffer) == n * rows * columns
                rawdata = np.frombuffer(rawbuffer, dtype='>u1', count=n * rows * columns)
                return rawdata.reshape(n, rows, columns).astype(np.float32) / 255.0

            def read_labels_archive(fi):
                magic, n = struct.unpack(">II", fi.read(8))
                assert magic == 0x00000801
                rawbuffer = fi.read()
                assert len(rawbuffer) == n
                return np.frombuffer(rawbuffer, dtype='>u1', count=n)

            # %%
            with gzip.open("train-images-idx3-ubyte.gz", "rb") as fp:
                train_images = read_images_archive(fp)
                print(f"Loaded training images with shape {train_images.shape}!")

            with gzip.open("train-labels-idx1-ubyte.gz", "rb") as fp:
                train_labels = read_labels_archive(fp)
                print(f"Loaded training labels with shape {train_labels.shape}!")

            with gzip.open("t10k-images-idx3-ubyte.gz", "rb") as fp:
                test_images = read_images_archive(fp)
                print(f"Loaded test images with shape {test_images.shape}!")

            with gzip.open("t10k-labels-idx1-ubyte.gz", "rb") as fp:
                test_labels = read_labels_archive(fp)
                print(f"Loaded test labels with shape {test_labels.shape}!")
            # %%
            # NOTE: visualize training dataset
            _nrows = 3
            fig, axes = plt.subplots(nrows=_nrows, ncols=10, figsize=(20, 5))
            for lbl in range(10):
                for r in range(_nrows):
                    ax = axes[r][lbl]
                    ax.imshow(train_images[train_labels == lbl][r], cmap='gray')
                    ax.xaxis.set_tick_params(labelbottom=False)
                    ax.yaxis.set_tick_params(labelleft=False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if r == 0:
                        ax.set_title(f"{lbl}")
            plt.show()

            # %%
            class MNISTDataset(Dataset):
                def __init__(self, images, labels) -> None:
                    # Check if the number of images matches the number of labels
                    if len(images) != len(labels):
                        raise ValueError(f"Different number of images ({len(images)}) and labels ({len(labels)})!")

                    # Store images and labels
                    self.images = images
                    self.labels = labels

                def __len__(self):
                    # Return the total number of samples in the dataset
                    return len(self.images)

                def __getitem__(self, index):
                    # Normalize the image data to a range between 0 and 1
                    image = self.images[index]  # [0 .. 1]

                    # For convolutional models: uncomment the next line
                    # image = image[None, ...]

                    # For linear models: flatten the image
                    image = image.flatten()

                    # Retrieve the label for the current index
                    label = self.labels[index]

                    # Convert the image and label to PyTorch tensors
                    image = torch.from_numpy(image)  # Convert to float32 tensor
                    label = torch.tensor(label, dtype=torch.long)  # Convert to long tensor

                    return image, label

            # %%
            class MLP(nn.Module):
                def __init__(self, nin: int, nouts: List[int]) -> None:
                    # Initialize the MLP module
                    super().__init__()

                    # Define the sizes of the layers
                    sizes = [nin] + nouts

                    # Create a list of linear layers
                    self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(nouts))])

                    # Store the number of layers
                    self.n_layers = len(self.layers)

                def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
                    # Forward pass through the MLP
                    for i, layer in enumerate(self.layers):
                        x = layer(x)  # Apply the linear transformation

                        # Apply activation function except for the last layer
                        if i != self.n_layers - 1:
                            x = F.relu(x)  # ReLU activation function

                    return x  # Return the output of the last layer

            # Function to initialize the parameters of a module
            def init_parameters(module):
                # Check if the module is an instance of nn.Linear
                if isinstance(module, nn.Linear):
                    # Initialize the weights of the linear module with uniform distribution between -1 and 1
                    module.weight = torch.nn.init.uniform_(module.weight, -1, 1)

                    # Initialize the bias of the linear module with zeros
                    module.bias.data.fill_(0.0)

            # %%
            def train_one_epoch(
                    model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: str = "cpu",
                    verbose: bool = True,
            ) -> dict:
                # Set the model to training mode
                model.train()

                # Lists to store losses and accuracies during training
                losses = []
                accuracies = []

                # tqdm provides a progress bar during training
                with tqdm(total=len(loader), desc="training", file=sys.stdout, ncols=100,
                          disable=not verbose) as progress:
                    for x_batch, y_true in loader:
                        # Move the input and target tensors to the specified device (CPU or GPU)
                        x_batch = x_batch.to(device)
                        y_true = y_true.squeeze(1).to(device)

                        # Zero out the gradients before backpropagation.
                        optimizer.zero_grad()

                        # Obtain model predictions
                        y_pred = model(x_batch)

                        # Compute the loss using the specified criterion.
                        loss = criterion(y_pred, y_true)

                        # Perform backward propagation to compute gradients
                        loss.backward()

                        # Update the model parameters using the optimizer
                        optimizer.step()

                        losses.append(loss.item())
                        accuracies.append((y_pred.argmax(1) == y_true).float().detach().cpu().numpy())

                        progress.set_postfix_str(f"loss {losses[-1]:.4f}")
                        progress.update(1)

                # Aggregate and return training logs
                logs = {
                    "losses": np.array(losses),
                    "accuracies": np.concatenate(accuracies)
                }
                return logs

            # %%
            @torch.inference_mode()
            def evaluate(
                    model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    device: str = "cpu",
                    verbose: bool = True,
            ) -> dict:
                # Set the model to evaluation mode
                model.eval()

                # Lists to store losses and accuracies during evaluation
                losses = []
                accuracies = []

                # tqdm provides a progress bar during evaluation
                for x_batch, y_true in tqdm(loader, desc="evaluation", file=sys.stdout, ncols=100, disable=not verbose):
                    # Move the input and target tensors to the specified device (CPU or GPU)
                    x_batch = x_batch.to(device)
                    y_true = y_true.squeeze(1).to(device)

                    # Obtain model predictions
                    y_pred = model(x_batch)

                    # Compute the loss using the specified criterion
                    loss = criterion(y_pred, y_true)

                    # Append the loss and accuracy for logging purposes
                    losses.append(loss.item())
                    accuracies.append((y_pred.argmax(1) == y_true).float().detach().cpu().numpy())

                # Aggregate and return evaluation logs
                logs = {
                    "losses": np.array(losses),
                    "accuracies": np.concatenate(accuracies)
                }
                return logs

# Check if CUDA (GPU) is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device - {device}")

# TODO: experiment with architectures
# Create a Multilayer Perceptron (MLP) model with a modified architecture
# In this architecture, you can adjust the number of neurons in the hidden layers or add more hidden layers
model = MLP(nin=28 * 28, nouts=[512, 256, 128, 10])  # Modify the architecture as needed

# MOVE MODEL TO DEVICE
# Move the model to the device to speed up training using GPU parallelization
model = model.to(device)

# SET RANDOM SEED FOR REPRODUCIBILITY
# Set a random seed to ensure reproducibility before initializing model parameters
torch.manual_seed(42)

# INITIALIZE MODEL PARAMETERS
# Initialize the model parameters
model.apply(init_parameters)

# PRINT THE NUMBER OF PARAMETERS
# Print the number of trainable parameters in the model
print("Number of trainable parameters -", sum(p.numel() for p in model.parameters() if p.requires_grad))

# CONFIGURE THE OPTIMIZER
# You can experiment with the learning rate to increase convergence
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# NOTE: you can also use a different optimizer to see its effect on optimization compared with SGD
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# NOTE: you can experiment with the batch size to see its effect on model convergence
batch_size = 64

# Create the training data loader
train_loader = DataLoader(
    MNISTDataset(train_images, train_labels),  # MNIST training dataset
    batch_size=batch_size,  # Set the batch size
    shuffle=True,  # Shuffle the data during training
    num_workers=os.cpu_count(),  # Number of CPU cores to use for data loading
    drop_last=True,  # Drop the last incomplete batch if its size is less than batch_size
)

# Create the validation data loader
valid_loader = DataLoader(
    MNISTDataset(test_images, test_labels),  # MNIST test dataset for validation
    batch_size=batch_size,  # Set the batch size
    shuffle=False,  # Do not shuffle the data during validation
    num_workers=os.cpu_count(),  # Number of CPU cores to use for data loading
    drop_last=False,  # Keep the last batch even if its size is less than batch_size
)

# NOTE: you can experiment with the number of epochs to train the model longer or shorter
n_epochs = 100

# Lists to store training and validation losses and accuracies for each epoch
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

# Loop over epochs for training
for ep in range(n_epochs):
    print(f"\nEpoch {ep + 1:2d}/{n_epochs:2d}")

    # Train the model for one epoch and collect training logs
    train_logs = train_one_epoch(model, train_loader, loss_fn, optimizer, device, verbose=True)
    train_losses.append(np.mean(train_logs["losses"]))
    train_accuracies.append(np.mean(train_logs["accuracies"]))
    print("      loss:", train_losses[-1])
    print("  accuracy:", train_accuracies[-1])

    # Evaluate the model on the validation set and collect validation logs
    valid_logs = evaluate(model, valid_loader, loss_fn, device, verbose=True)
    valid_losses.append(np.mean(valid_logs["losses"]))
    valid_accuracies.append(np.mean(valid_logs["accuracies"]))
    print("      loss:", valid_losses[-1])
    print("  accuracy:", valid_accuracies[-1])

# Plotting training and validation performance for comparison
fig, axes = plt.subplots(ncols=2, figsize=(15, 4))

# Plot training and validation losses
axes[0].plot(np.arange(len(train_losses)), train_losses, ".-")
axes[0].plot(np.arange(len(valid_losses)), valid_losses, ".-")
axes[0].legend(["Train", "Validation"])
axes[0].set_title("Loss")  # Set title for the loss subplot
axes[0].grid()  # Add grid lines to the plot

# Plot training and validation accuracies
axes[1].plot(np.arange(len(train_accuracies)), train_accuracies, ".-")
axes[1].plot(np.arange(len(valid_accuracies)), valid_accuracies, ".-")
axes[1].legend(["Train", "Validation"])
axes[1].set_title("Accuracy")  # Set title for the accuracy subplot
axes[1].grid()  # Add grid lines to the plot

# Show the plots
fig.tight_layout()
plt.show()
