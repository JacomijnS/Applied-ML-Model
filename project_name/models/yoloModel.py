from ultralytics import YOLO


class YOLOModel(YOLO):
    def __init__(self, model_path: str = "yolo11n-obb.pt"):
        """
        Initialize the YOLO model with the specified path.

        :param model_path: Path to the YOLO model file.
        """
        super().__init__(model_path)

    def train(self, **kwargs):
        """
        Train the YOLO model on the specified dataset.

        :param data_path: Path to the dataset configuration file.
        :param epochs: Number of training epochs.
        """
        super().train(
            **kwargs
        )

    def load_model(self, model_path: str):
        """
        Load a YOLO model from a specified path.

        :param model_path: Path to the YOLO model file.
        """
        self.__init__(model_path)

    def predict(self, **kwargs):
        """
        Perform inference on an image and return the results.
        """
        return super().predict(**kwargs)
