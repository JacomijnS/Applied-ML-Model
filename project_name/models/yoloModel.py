from ultralytics import YOLO


class YOLOModel(YOLO):
    def __init__(self, model_path: str = "yolo11n-obb.pt"):
        """
        Initialize the YOLO model with the specified path.

        :param model_path: Path to the YOLO model file.
        """
        super().__init__(model_path)

    def train(self, pathToData: str, epochs: int = 100, batch: int = 8):
        """
        Train the YOLO model on the specified dataset.

        :param data_path: Path to the dataset configuration file.
        :param epochs: Number of training epochs.
        """
        super().train(
            data=pathToData,
            epochs=epochs,
            batch=batch,
            # Data augmentation parameters
            degrees=180,    # We can rotate the images between -180 and 180
                            # degrees because x-ray images don't have a
                            # specific orientation
            flipud=0.5,  # There is a 50% chance to flip the image upside down,
                         # because x-ray images don't have a specific
                         # orientation
            fliplr=0.5,  # There is a 50% chance to flip the image
                         # left to right, because x-ray images don't have a
                         # specific orientation
            hsv_v=0.4,  # The image intensity is randomly selected between
                        # -0.4 and 0.4 because x-ray images can have different
            device=0,   # intensities for GPU training
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
        return super().__call__(**kwargs)
