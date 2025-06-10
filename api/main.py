from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
from project_name.models.yoloModel import YOLOModel
from PIL import Image
import io
import json
from pydantic import BaseModel


sys.path.append('..')
app = FastAPI()

# Allow for CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLOModel()
model.load_model("runs/obb/train6/weights/best.pt")

class Bbox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float

class PredictionResponse(BaseModel):
    filename: str
    name: str
    confidence: float
    bbox: Bbox | None


predictions = {}


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


# Handle file uploads and predictions
@app.post(
    "/predict",
    description="Endpoint to upload an image and get predictions.",
    response_model=PredictionResponse
)
async def predict(file: UploadFile = File(
                ...,
                description= "Upload an image for prediction, accepted formats are .jpg, .jpeg, and .png.")):
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
        json_str = results[0].to_json()
        json_data = json.loads(json_str)

        # if no fracture was found
        if not json_data: 
            response_obj = PredictionResponse(
                filename=file.filename,
                name="no fracture",
                confidence=0.0,
                bbox=None
            )
            predictions[file.filename] = response_obj
            return response_obj

        pred = json_data[0]
        
        # Transform json to an object
        response_obj = PredictionResponse(
            filename=file.filename,
            name=pred["name"],
            confidence=pred["confidence"],
            bbox=Bbox(**pred["box"])
        )

        predictions[file.filename] = response_obj
        print(results)
        return response_obj
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only .jpg files are allowed."
        )
