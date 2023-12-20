import json
import cv2
import numpy as np
from joblib import load

def generate_histogram(image_path, color_space='RGB'):
    # Read the image
    image = cv2.imread(image_path)
    
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

def predict_label(image_path, model_filename, color_space='RGB'):
    # Load the trained model
    classifier = load(model_filename)
    
    histogram = generate_histogram(image_path, color_space)
    prediction = classifier.predict([histogram])
    return prediction[0]

def predict_top_n_labels(image_path, model_filename, color_space='RGB', top_n=3):
    # Load the trained model
    classifier = load(model_filename)
    
    # Generate histogram for the image
    histogram = generate_histogram(image_path, color_space)
    
    # Get probabilities
    probabilities = classifier.predict_proba([histogram])[0]
    
    # Get class labels
    class_labels = classifier.classes_
    
    # Get top N predictions
    top_n_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_n_predictions = [(class_labels[i], probabilities[i]) for i in top_n_indices]

    return top_n_predictions

def predict_top_n_labels_json(image_path, model_filename, color_space='RGB', top_n=3):
    # Load the trained model
    classifier = load(model_filename)
    
    # Generate histogram for the image
    histogram = generate_histogram(image_path, color_space)
    
    # Get probabilities
    probabilities = classifier.predict_proba([histogram])[0]
    
    # Get class labels
    class_labels = classifier.classes_
    
    # Get top N predictions
    top_n_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_n_predictions = [{"label": class_labels[i], "probability": float(probabilities[i])} for i in top_n_indices]

    return json.dumps(top_n_predictions, indent=4)

model_filename = 'classifier_HSL.joblib' # seems the best mode so far

# Example usage for regular single prediction
# predicted_label = predict_label('test/20231218_204508.png', model_filename)
# print("Predicted Label: ", predicted_label)

# Example usage for top N prediction
# top_predictions = predict_top_n_labels('test/20231218_204415.png', model_filename, top_n=3)
# for label, prob in top_predictions:
#     print(f"Label: {label}, Probability: {prob}")

# Example usage for top N prediction with JSON output
json_predictions = predict_top_n_labels_json('test/20231218_204415.png', model_filename, top_n=3)
print(json_predictions)