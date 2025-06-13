import torch
import torch.nn.functional as F
from torch import nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class SlidingWindowCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(SlidingWindowCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.fc1 = nn.Linear(16 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

def generate_patches(img, window_size=32, stride=8):
    patches = []
    coords = []
    height, width = img.shape
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            patch = img[y:y + window_size, x:x + window_size]
            patches.append(patch)
            coords.append((x, y, x + window_size, y + window_size))
    return np.array(patches), coords

def make_heatmap_for_image(
    image_path,
    model_path="sliding_window_cnn.pt",
    window_size=32,
    stride=8,
    show=True,
    save_path=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
    # Patchify
    patches, coords = generate_patches(img, window_size, stride)
    # Load model
    model = SlidingWindowCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # Predict
    with torch.no_grad():
        batch_patches = np.expand_dims(patches, axis=1)  # (N, 1, H, W)
        batch_patches_tensor = torch.tensor(batch_patches, dtype=torch.float32).to(device)
        outputs = model(batch_patches_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
    # Build heatmap
    heatmap = np.zeros(img.shape, dtype=np.float32)
    heatmap_counts = np.zeros(img.shape, dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(coords):
        prob = probs[i, 1]  # positive class probability
        heatmap[y1:y2, x1:x2] += prob
        heatmap_counts[y1:y2, x1:x2] += 1
    heatmap_counts[heatmap_counts == 0] = 1
    avg_heatmap = heatmap / heatmap_counts
    # Show/save result
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.imshow(avg_heatmap, cmap='jet', alpha=0.5)
    plt.colorbar(label='Probability (fracture)')
    plt.title(f"Fracture Probability Heatmap\n{os.path.basename(image_path)}")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--model_path", type=str, default="sliding_window_cnn.pt", help="Path to trained model weights")
    parser.add_argument("--window_size", type=int, default=32, help="Patch window size")
    parser.add_argument("--stride", type=int, default=8, help="Stride for patch extraction")
    parser.add_argument("--no_show", action='store_true', help="Do not display the image")
    parser.add_argument("--save_path", type=str, default=None, help="Save heatmap to file")
    args = parser.parse_args()
    make_heatmap_for_image(
        args.image_path,
        model_path=args.model_path,
        window_size=args.window_size,
        stride=args.stride,
        show=not args.no_show,
        save_path=args.save_path
    )
