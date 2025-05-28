from ultralytics import YOLO


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
