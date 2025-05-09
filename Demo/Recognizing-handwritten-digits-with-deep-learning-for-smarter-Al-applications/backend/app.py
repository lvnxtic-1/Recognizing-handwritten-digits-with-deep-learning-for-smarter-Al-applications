from flask import Flask, render_template, request, jsonify
from model import predict_digit
import numpy as np
import base64
import cv2
import io
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return 'Digit Recognition API is Running'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file).convert('L').resize((28, 28))
    image_array = np.array(image)

    digit, confidence = predict_digit(image_array)
    return jsonify({'prediction': int(digit), 'confidence': round(confidence, 2)})

if __name__ == '__main__':
    app.run(debug=True)
