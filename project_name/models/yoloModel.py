from ultralytics import YOLO

import cv2
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator


class YOLOModel(YOLO):
    def __init__(self):
        """
        Initialize the YOLO model with the specified path.

        :param model_path: Path to the YOLO model file.
        """
        super().__init__()
        # We load a pretrained YOLO model
        self.model = YOLO("yolo11n-obb.pt")

    def train(self, pathToData: str, epochs: int = 100, batch: int = 16):
        """
        Train the YOLO model on the specified dataset.

        :param data_path: Path to the dataset configuration file.
        :param epochs: Number of training epochs.
        """
        self.model.train(
            data=pathToData,
            epochs=epochs,
            batch=batch,
            obb=True  # Enable Oriented Bounding Boxes
        )

    def load_model(self, model_path: str):
        """
        Load a YOLO model from a specified path.

        :param model_path: Path to the YOLO model file.
        """
        self.model = YOLO(model_path)

    def predict(self, **kwargs):
        """
        Perform inference on an image and return the results.

        :param image_path: Path to the image file.
        :return: Inference results.
        """
        results = self.model(**kwargs)
        return results


def plot_preds_vs_truth(image_path: str, results, label_path: str):
    image = cv2.imread(image_path)
    annotator = Annotator(image.copy(), example='label')

    # Draw predictions
    for box in results[0].obb.xyxy.cpu().numpy():
        annotator.box_label(box, label="pred", color=(0, 255, 0))

    # Draw ground truth (YOLO OBB format: class cx cy w h angle)
    with open(label_path, 'r') as f:
        for line in f:
            cls, cx, cy, w, h, angle = map(float, line.strip().split())
            # Convert rotated box to rectangle manually or using helper
            # This is simplified — YOLOv8 may expose helper utils soon

    plt.figure(figsize=(10, 10))
    plt.imshow(annotator.im[..., ::-1])
    plt.title("Prediction (Green) vs Ground Truth")
    plt.axis('off')
    plt.show()
