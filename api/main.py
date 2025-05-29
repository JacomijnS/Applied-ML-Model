from fastapi import FastAPI, File, UploadFile, HTTPException
import sys
sys.path.append("..")  # Adjust the path to your project structure
from project_name.models.yoloModel import YOLOModel
from PIL import Image
import io

app = FastAPI()
model = YOLOModel()
model.load_model("../runs/obb/train2/weights/best.pt")

predictions = {}

@app.post("/predict")
async def predict(file: UploadFile):
    if file.filename == "" or file is None:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    if file.filename in predictions:
        print(f"Returning cached prediction for {file.filename}")
        return predictions[file.filename]
    if file.filename.endswith(('.jpg', '.jpeg', '.png')):
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Predict on an example image (adjust path if needed)
        results = model.predict(
            source=image,
            save=False
        )
        predictions[file.filename] = results
        for r in results:
            r.show()  # Show the results in a window
        return results
    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Only .jpg files are allowed.")