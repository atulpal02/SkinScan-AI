from fastapi import FastAPI, UploadFile, File, Body
import numpy as np
import cv2
import base64

from preprocessing.preprocess import preprocess
from inference.predict import predict_image
from rag.explain import generate_explanation, answer_question

app = FastAPI()

# ✅ PREDICT ROUTE
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_image, info = preprocess(image)
    result = predict_image(processed_image)

    _, buffer = cv2.imencode('.jpg', processed_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    _, buffer = cv2.imencode('.jpg', result["heatmap"])
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

    explanation = generate_explanation(result["prediction"])

    return {
        "result": {
            "prediction": result["prediction"],
            "confidence": result["confidence"]
        },
        "explanation": explanation,
        "enhanced_image": img_base64,
        "heatmap": heatmap_base64
    }


# ✅ ASK ROUTE (OUTSIDE)
@app.post("/ask")
async def ask_question(data: dict = Body(...)):
    prediction = data.get("prediction")
    question = data.get("question")

    answer = answer_question(prediction, question)

    return {
        "answer": answer
    }