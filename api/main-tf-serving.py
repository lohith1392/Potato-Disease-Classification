from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint ="http://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAMES = ["EARLY BLIGHT", "LATE BLIGHT", "HEALTHY"]


@app.get("/ping")
async def ping():
    return "Hello , I am alive"


def read_file_as_image(data) -> np.ndarray:
    value = Image.open(BytesIO(data))
    image = np.array(value)
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    image_batch = np.expand_dims(image, 0)

    json_data ={
        "instances": image_batch.tolist()
    }
    response = requests.post(endpoint,json=json_data)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
