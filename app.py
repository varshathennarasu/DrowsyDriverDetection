from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from pygame import mixer
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Initialize pygame mixer for sound
mixer.init()
mixer.music.load('music.wav')  # Ensure 'music.wav' is in the same directory

# Load the trained model (for image classification)
MODEL_PATH = "drowsiness_model.h5"
model = load_model(MODEL_PATH)
print("Model loaded successfully.")


# Function to preprocess image for classification
def preprocess_image(image):
    img = image.resize((64, 64))  # Resize to match model input size
    img = img.convert("L")  # Convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 64, 64, 1)  # Reshape to match model input shape
    return img_array


@app.route('/')
def index():
    return render_template('index.html')  # This renders the index.html page


@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        if file:
            image = Image.open(file)  # Open the uploaded image file
            processed_image = preprocess_image(image)  # Preprocess image
            print("Image preprocessed successfully.")

            prediction = model.predict(processed_image)  # Get model prediction
            print("Prediction result:", prediction)

            # Drowsy driver if the first class (Closed Eyes) is more likely, else alert driver
            if prediction[0][0] > prediction[0][1]:
                result = "Drowsy Driver: Closed Eyes"
                mixer.music.play()  # Play alert sound if closed eyes detected
            else:
                result = "Alert Driver: Open Eyes"

            return jsonify({"prediction": result})

        else:
            return jsonify({"error": "No file uploaded"}), 400

    except Exception as e:
        print(f"Error processing image: {str(e)}")  # Log the error
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)





