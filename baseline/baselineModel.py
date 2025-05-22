# Most of this is based of https://medium.com/@myringoleMLGOD/simple-convolutional
# -neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import cv2
import numpy as np


class SlidingWindowCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2): 
    # We only want if there is a fracture or not 
    # (maybe also add if it can see which place it is, but no clue how to do multiple)
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For our x-rays, this is 1 (grayscale images).
            num_classes: int
                The number of classes we want to predict, in our case this is 2
        """
        super(SlidingWindowCNN, self).__init__()

        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Max pooling layer: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Fully connected layer: 64 -> 32 -> 16 input features (after two 2x2 poolings), 2 output features (num_classes)
        self.fc1 = nn.Linear(16 * 16 * 16, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        return x

# Now we need to make a mechanism that separates the image into patches

# Generate patches for each
def generate_patches(img, window_size=64, stride=32):
    patches = []
    coords = []
    height, width = img.shape   # Should always be 512 by 512
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            patch = img[y:y + window_size, x:x + window_size]
            patches.append(patch)
            coords.append((x, y, x + window_size, y + window_size))
    return np.array(patches), coords


class PatchDataset():
    def __init__(self, image_direction, window_size=64, stride=32): # Window is 64x64
        self.image_paths = [
            os.path.join(image_direction, f) for f in os.listdir(image_direction).endswith('.jpg')
            ] # Sanity check for images
        self.window_size = window_size
        self.stride = stride
        self.samples = []

        for path in self.image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
            patches, coords = generate_patches(img, window_size, stride)
            for p in patches:
                self.samples.append(p)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch = self.samples[idx]
        patch = np.expand_dims(patch, axis=0)  # add channel
        return torch.tensor(patch, dtype=torch.long), 0  # Still need to add labels

# TRaining process
def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SlidingWindowCNN().to(device)

    num_epochs = 10  # Reduced for demonstration purposes
    batch_size = 64
    learning_rate = 0.001

    train_dataset = datasets.PatchDataset("data/train/images", window_size=64, stride=32)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):

        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass: compute the model output
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # Optimization step: update the model parameters
            optimizer.step()