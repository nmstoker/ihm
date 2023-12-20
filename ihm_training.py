import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

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

def load_images_from_folder(base_folder):
    file_paths = []
    labels = []

    for label_folder in os.listdir(base_folder):
        label_folder_path = os.path.join(base_folder, label_folder)
        if os.path.isdir(label_folder_path):
            for file in os.listdir(label_folder_path):
                if file.endswith('.jpg') or file.endswith('.png'):  # Add more file types if needed
                    file_paths.append(os.path.join(label_folder_path, file))
                    labels.append(label_folder)
    
    return file_paths, labels

base_folder = '/datasets/images'
file_paths, labels = load_images_from_folder(base_folder)

color_space="HSL" # RGB | HSL | HSV

# Generate histograms
histograms = [generate_histogram(file, color_space=color_space) for file in file_paths]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(histograms, labels, test_size=0.2, random_state=42)

# Train the classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save the model
model_filename = f'classifier_{color_space}.joblib'
dump(clf, model_filename)

# Evaluate the classifier
print("Model Accuracy: ", clf.score(X_test, y_test))
