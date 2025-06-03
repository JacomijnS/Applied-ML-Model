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

    def train(self, pathToData: str, epochs: int = 100, batch: int = 8):
        """
        Train the YOLO model on the specified dataset.

        :param data_path: Path to the dataset configuration file.
        :param epochs: Number of training epochs.
        """
        self.model.train(
            data=pathToData,
            epochs=epochs,
            batch=batch,
            # Data augmentation parameters
            degrees = 180, # We can rotate the images between -180 and 180 degrees because x-ray images don't have a specific orientation
            flipud=0.5, # There is a 50% chance to flip the image upside down, because x-ray images don't have a specific orientation
            fliplr=0.5, # There is a 50% chance to flip the image left to right, because x-ray images don't have a specific orientation
            hsv_v=0.4, # The image intensity is randomly selected between -0.4 and 0.4 because x-ray images can have different intensities
            device=0,   # for GPU training
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
