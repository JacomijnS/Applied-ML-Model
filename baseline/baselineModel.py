# Most of this is based of https://medium.com/@myringoleMLGOD/simple-convolutional
# -neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80

import torch
import torch.nn.functional as F
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


# Calculate Intersection of union, this function is taken from the following guide:
# https://www.v7labs.com/blog/intersection-over-union-guide
def compute_iou(box1, box2):
    # Calculate intersection area
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2
    intersection_width = min(r1, r2) - max(l1, l2)
    intersection_height = min(b1, b2) - max(t1, t2)
    
    if intersection_width <= 0 or intersection_height <= 0:
        return 0
    
    intersection_area = intersection_width * intersection_height

    # Calculate union area
    box1_area = (r1 - l1) * (b1 - t1)
    box2_area = (r2 - l2) * (b2 - t2)
    
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

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
    

    def __init__(self, image_directory, label_directory, window_size=64, stride=32): # Window is 64x64
        
        # Transform the ploygon to box
        def polygon_to_bbox(coords):
            xs = coords[::2]
            ys = coords[1::2]
            return min(xs), min(ys), max(xs), max(ys)
        
        self.image_paths = [
            os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith('.jpg')
        ]
        self.window_size = window_size
        self.stride = stride
        self.samples = []

        for path in self.image_paths:
            # Get filename and get the correct label
            filename = os.path.basename(path)
            label_path = os.path.join(label_directory, filename.replace('.jpg', '.txt'))

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
            patches, coords = generate_patches(img, window_size, stride)
            
            boxes = [] # Save boxes
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        # ignore class ts
                        poly = list(map(float, parts[1:]))
                        # group to x,y pairs, denormalize to pixel coords
                        xs = poly[::2]
                        ys = poly[1::2]
                        xs = [int(x * img.shape[1]) for x in xs]
                        ys = [int(y * img.shape[0]) for y in ys]
                        bbox = polygon_to_bbox([*xs, *ys])
                        boxes.append(bbox)

            for p in range(len(patches)):
                label = 0
                for box in boxes:
                    if compute_iou(box, coords[p]) > 0.2: # if more than half is 
                        label = 1
                        break
                self.samples.append((patches[p], label)) # store patch with label as a double


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, label = self.samples[idx]
        patch = np.expand_dims(patch, axis=0)  # add channel
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# TRaining process
def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SlidingWindowCNN().to(device)

    num_epochs = 10  # Reduced for demonstration purposes
    batch_size = 64
    learning_rate = 0.001

    train_dataset = PatchDataset("data/train/images", "data/train/labels", window_size=64, stride=32)

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
    
    torch.save(model.state_dict(), "sliding_window_cnn.pt")
    print("Model saved to sliding_window_cnn.pt")


if __name__ == "__main__":
    train_model()