from project_name.models.yoloModel import YOLOModel
import yaml

TRAIN = False
PREDICT = False
VALIDATE = False
TUNE = True
FINALTRAIN = False
# Subset for testing purposes
dataPath = "project_name/data/data.yaml"
# We grab one of the images from test set
testPath = (
    "project_name/data/test/images"
)
testLabelPath = (
    "project_name/data/test/labels"
)

# Instantiate your model class
model = YOLOModel()

# Train it (run 1 epoch just for testing)
if TRAIN is True:
    # Train the model on the subset of data
    print("Training the model on the subset of data...")
    model.train(pathToData=dataPath, epochs=60)

if PREDICT is True:
    print("Predicting on an example image...")

    # Load the trained model
    model.load_model("runs/obb/train2/weights/best.pt")
    print("Model loaded successfully.")
    # Predict on an example image (adjust path if needed)
    results = model.predict(
        source=testPath,
        save=False
    )
    for r in results:
        r.show()  # Show the results in a window

if VALIDATE is True:
    print("Validating the model on the validation set...")

    # Load the trained model
    model.load_model("runs/obb/train8/weights/best.pt")
    results = model.val(data=dataPath, plots=True)
    print(results.confusion_matrix.to_df())

if TUNE is True:
    print("Tuning the model on the validation set...")

    # Load the trained model
    model.load_model("runs/obb/train6/weights/best.pt")
    results = model.tune(
        data=dataPath,
        epochs=50,
        batch= 8,
    )

if FINALTRAIN is True:
    print("Final training on the dataset...")

    # Load the hyperparameters from the tuning results
    with open("runs/detect/tune/best_hyperparameters.yaml", "r") as f:
        best_hyp = yaml.safe_load(f)

    model.train(data="path/to/data.yaml", epochs=100, **best_hyp)

    print("Final training completed.")
