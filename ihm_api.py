from fastapi import FastAPI, File, UploadFile, Form
import cv2
import numpy as np
import json
from joblib import load

app = FastAPI()

def generate_histogram(image, color_space='RGB'):
    # Convert color space if needed
    if color_space == 'HSL':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    elif color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram
    histogram = []
    for i in range(3):  # Assuming a 3-channel image
        channel_hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histogram.extend(channel_hist.flatten())

    return np.array(histogram)

def predict_top_n_labels_json(image, model_filename, color_space='RGB', top_n=3):
    # Load the trained model
    classifier = load(model_filename)

    # Generate histogram for the image
    histogram = generate_histogram(image, color_space)

    # Get probabilities
    probabilities = classifier.predict_proba([histogram])[0]

    # Get class labels
    class_labels = classifier.classes_

    # Get top N predictions
    top_n_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_n_predictions = [{"label": class_labels[i], "probability": float(probabilities[i])} for i in top_n_indices]

    return json.dumps(top_n_predictions, indent=4)

@app.post("/predict/")
async def create_upload_file(color_space: str = Form(...), top_n: int = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid or corrupt image file"}

    model_filename = f'classifier_{color_space}.joblib'
    json_predictions = predict_top_n_labels_json(image, model_filename, color_space, top_n)
    return {"predictions": json_predictions}
