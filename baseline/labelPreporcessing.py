import os
import cv2


def transform_label(norm_x, norm_y, orig_w, orig_h, target_size=512):
    # Denormalize
    x = norm_x * orig_w
    y = norm_y * orig_h
    # compute scale and padding (as in preprocessing.py)
    scale = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    # Scale and add padding
    x_resized = x * scale + pad_left
    y_resized = y * scale + pad_top
    return x_resized, y_resized


input_label_dir = "../project_name/data/train/labels"
input_image_dir = "../project_name/data/train/images"
output_label_dir = "data/train/labels"

# take text and put it into the data repo for this data
for filename in os.listdir(input_label_dir):
    if not filename.lower().endswith('.txt'):
        continue

    label_path = os.path.join(input_label_dir, filename)
    image_name = filename.replace('.txt', '.jpg')
    image_path = os.path.join(input_image_dir, image_name)
    output_path = os.path.join(output_label_dir, filename)

    if not os.path.exists(image_path):
        print(f"Missing image for {filename}")
        continue

    # Load the original image to get original shape
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = img.shape

    with open(label_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            class_id = parts[0]
            norm_coords = list(map(float, parts[1:]))

            # Transform every x, y pair
            transformed_coords = []
            for i in range(0, len(norm_coords), 2):
                x, y = norm_coords[i], norm_coords[i + 1]
                x_new, y_new = transform_label(x, y, orig_w, orig_h)
                # Save as normalized coordinates in the new 512x512 image
                x_new_norm = x_new / 512
                y_new_norm = y_new / 512
                transformed_coords.extend([x_new_norm, y_new_norm])

            # Write to file in the same format
            outfile.write(f"{class_id} " + " ".join(f"{v:.8f}" for v in transformed_coords) + "\n")