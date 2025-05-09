import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "digit_recognition_model.h5"

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize and reshape
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]  # Shape: (28, 28, 1)
    x_test = x_test[..., np.newaxis]
    return x_train, y_train, x_test, y_test

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model():
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=2)
    model.save(MODEL_PATH)
    print("✅ Model trained and saved to:", MODEL_PATH)

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return load_model(MODEL_PATH)

def predict_digit(image_array):
    """
    Expects a 28x28 grayscale image as a NumPy array.
    Returns predicted digit and confidence.
    """
    model = load_trained_model()
    image_array = image_array.reshape(1, 28, 28, 1).astype("float32") / 255.0
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence
