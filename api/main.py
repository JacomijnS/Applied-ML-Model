from fastapi import FastAPI, UploadFile, HTTPException
import sys
sys.path.append('..')
from project_name.models.yoloModel import YOLOModel
from PIL import Image
import io
import json
from pydantic import BaseModel

app = FastAPI()
model = YOLOModel()
model.load_model("../runs/obb/train6/weights/best.pt")

class PredictionResponse(BaseModel):
    filename: str
    name: str
    confidence: float
    bbox: list

predictions = {}
# Handle file uploads and predictions
@app.post("/predict", description="Endpoint to upload an image and get predictions.", response_model=PredictionResponse)
async def predict(file: UploadFile):
    # Check if the file is empty or None
    if file.filename == "" or file is None:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    # Check if the file is already in predictions cache
    if file.filename in predictions:
        print(f"Returning cached prediction for {file.filename}")
        return predictions[file.filename]
    # Check if the file is an image
    if file.filename.endswith(('.jpg', '.jpeg', '.png')):
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        results = model.predict(
            source=image,
            save=False
        )
        json_str = results[0].tojson()
        json_data = json.loads(json_str)
        predictions[file.filename] = json_data
        print(results)
        return json_data
    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Only .jpg files are allowed.")