
import torch
import torch.nn.functional as F
from torch import nn
import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


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
        self.fc1 = nn.Linear(16 * 8 * 8, num_classes)

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

    def __init__(self, image_directory, label_directory, window_size=64, stride=32, negative_to_positive_ratio=1.05): # Window is 64x64
        
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


        positive_count = 0
        negative_count = 0

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
                patch = patches[p]
                # Skip patch if it is nearly all padding (all zeros)
                if np.mean(patch) < 1e-3:  # Tweak this threshold if needed
                    continue

                label = 0
                for box in boxes:
                    if compute_iou(box, coords[p]) > 0.5:
                        label = 1
                        break

                # Balance: keep only up to ratio * positive_count negatives
                if label == 1:
                    self.samples.append((patch, label))
                    positive_count += 1
                elif negative_count < negative_to_positive_ratio * (positive_count + 1):
                    self.samples.append((patch, label))
                    negative_count += 1

        print(f"Loaded {len(self.samples)} patches: {positive_count} positive, {negative_count} negative.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, label = self.samples[idx]
        patch = np.expand_dims(patch, axis=0)  # add channel dimension for PyTorch: [C, H, W]
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    

def evaluate_fracture_detection():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SlidingWindowCNN().to(device)
    model.load_state_dict(torch.load("sliding_window_cnn.pt", map_location=device))
    model.eval()

    image_dir = "data/test/images"
    label_dir = "data/test/labels"
    window_size = 32
    stride = 8

    all_patch_targets = []
    all_patch_preds = []

    image_level_TP = 0
    image_level_FN = 0
    image_level_FP = 0
    image_level_TN = 0

    for filename in os.listdir(image_dir):
        if not filename.endswith('.jpg'):
            continue
        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
        patches, coords = generate_patches(img, window_size, stride)
        gt_boxes = []
        # Parse ground truth boxes
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    poly = list(map(float, parts[1:]))
                    xs = [x * img.shape[1] for x in poly[::2]]
                    ys = [y * img.shape[0] for y in poly[1::2]]
                    box = (min(xs), min(ys), max(xs), max(ys))
                    gt_boxes.append(box)

        # For each patch: does it overlap any gt box? (label 1 or 0)
        patch_labels = []
        for patch_idx in range(len(patches)):
            label = 0
            for box in gt_boxes:
                if compute_iou(box, coords[patch_idx]) > 0.5:
                    label = 1
                    break
            patch_labels.append(label)
        all_patch_targets.extend(patch_labels)

        # Predict for all patches
        predicted_labels = []
        with torch.no_grad():
            batch_patches = np.expand_dims(patches, axis=1)
            batch_patches_tensor = torch.tensor(batch_patches, dtype=torch.float32).to(device)
            outputs = model(batch_patches_tensor)
            probs = F.softmax(outputs, dim=1)
            predicted = (probs[:, 1] > 0.15).long().cpu().numpy().tolist()
            print("Max probability for positive class in this image:", probs[:, 1].max().item())
            print("Mean probability for positive class in this image:", probs[:, 1].mean().item())

            

        all_patch_preds.extend(predicted)

        # ---- IMAGE-LEVEL METRICS ----
        # GT: does this image have any fracture at all?
        gt_has_fracture = any([d == 1 for d in patch_labels])
        pred_has_fracture = any([k == 1 for k in predicted_labels])
        if gt_has_fracture and pred_has_fracture:
            image_level_TP += 1
        elif gt_has_fracture and not pred_has_fracture:
            image_level_FN += 1
        elif not gt_has_fracture and pred_has_fracture:
            image_level_FP += 1
        else:
            image_level_TN += 1

    # Patch-level metrics
    patch_precision = precision_score(all_patch_targets, all_patch_preds)
    patch_recall = recall_score(all_patch_targets, all_patch_preds)
    patch_f1 = f1_score(all_patch_targets, all_patch_preds)
    print("\nPatch-level metrics:")
    print(f"  Precision: {patch_precision:.4f}")
    print(f"  Recall:    {patch_recall:.4f}")
    print(f"  F1-score:  {patch_f1:.4f}")

    # Image-level metrics
    print("\nImage-level (fracture detection in whole image):")
    print(f"  TP (correct fracture detection): {image_level_TP}")
    print(f"  FN (missed fracture):            {image_level_FN}")
    print(f"  FP (false fracture alert):       {image_level_FP}")
    print(f"  TN (correct normal):             {image_level_TN}")
    if image_level_TP + image_level_FN > 0:
        sensitivity = image_level_TP / (image_level_TP + image_level_FN)
        print(f"  Sensitivity (recall for fracture images): {sensitivity:.4f}")
    if image_level_TN + image_level_FP > 0:
        specificity = image_level_TN / (image_level_TN + image_level_FP)
        print(f"  Specificity (normal images):            {specificity:.4f}")


if __name__ == "__main__":
    evaluate_fracture_detection()