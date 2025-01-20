
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model("drowsiness_model.h5")

# Function to test on a single image
def test_model_on_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    img = cv2.resize(img, (64, 64))  # Resize to match model input size
    img = img / 255.0  # Normalize the image
    img = img.reshape(1, 64, 64, 1)  # Reshape to match model input

    # Predict the state of the eyes
    prediction = model.predict(img)

    # Display the result (0: closed eyes, 1: open eyes)
    if prediction[0][0] > prediction[0][1]:
        print("Prediction: Closed Eyes")
    else:
        print("Prediction: Open Eyes")

# Test with a few sample images
test_model_on_image('./kaggle_data/train/Closed_Eyes/s0001_00001_0_0_0_0_0_01.png')
test_model_on_image('./kaggle_data/train/Open_Eyes/s0001_02334_0_0_1_0_0_01.png')
