# Recognizing Handwritten Digits with Deep Learning for Smarter AI Applications

This project demonstrates the use of Convolutional Neural Networks (CNNs) to recognize handwritten digits from the MNIST dataset. It includes the full machine learning pipeline—from model creation, training, and evaluation to deployment via a web interface. This solution showcases how deep learning can be applied in real-world AI applications, such as form recognition, postal code reading, and automated digit entry.

---

## 📝 Problem Statement

Recognizing handwritten digits (0–9) is a fundamental problem in the field of computer vision and machine learning. The task is to automatically classify a given image of a digit into one of the ten classes (0–9). This capability can be used in various real-world scenarios like:

* Automating data entry from handwritten forms.
* Digit recognition in postal services (ZIP code recognition).
* Recognizing numbers on checks or invoices.

This is a supervised, multi-class classification problem with a dataset of images of handwritten digits.

---

## 🎯 Project Objectives

* **Build a CNN model** that accurately classifies handwritten digits from the MNIST dataset.
* **Achieve over 98% accuracy** on unseen handwritten digit images.
* **Provide a user-friendly web interface** to allow anyone to upload a digit image and see the predicted result.
* **Deploy the model** using platforms like Gradio or Flask for easy access.

---

## 🧠 Technologies & Tools

* **Python 3.8+**
* **TensorFlow / Keras** (for model training and inference)
* **NumPy, Pandas, Matplotlib** (for data manipulation and visualization)
* **Flask** (for web interface, optional) or **Gradio** (for quicker deployment)
* **HTML, CSS** (for building the web frontend if using Flask)
* **Hugging Face Spaces** (for hosting the application)

---

## 📁 Project Structure

```
handwritten-digit-recognizer/
│
├── app.py                 ← Web application backend (Flask or Gradio)
├── model.py               ← CNN model architecture and loader
├── train_model.ipynb      ← Model training and evaluation notebook
├── mnist_cnn.h5           ← Trained model file (saved CNN model)
├── requirements.txt       ← Python dependencies
├── static/                ← Optional static files (e.g., sample images)
├── templates/             ← HTML frontend (Flask)
│   └── index.html
├── README.md              ← Project documentation
└── Procfile, runtime.txt  ← Deployment configurations (for Heroku or cloud services)
```

---

## 🚀 Setup Instructions

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
```

### 2. Create and activate a virtual environment (optional but recommended):

For **Windows**:

```bash
python -m venv venv
.\venv\Scripts\activate
```

For **Linux/macOS**:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the model training (if not using the pre-trained model):

Run the `train_model.ipynb` notebook to train the model and save the trained model as `mnist_cnn.h5`.

### 5. Run the app:

For **Flask** backend:

```bash
python app.py
```

For **Gradio** (if you are using Gradio for deployment):

```bash
python app_gradio.py
```

This will start the web application locally, and you can access it by visiting `http://localhost:5000` (for Flask) or the Gradio link in the console.

---

## 📊 Model Performance

* **Dataset**: MNIST (60,000 training images, 10,000 test images)
* **Model**: Convolutional Neural Network (CNN) with the following layers:

  * Conv2D → MaxPooling → Conv2D → Flatten → Dense → Softmax
* **Test Accuracy**: \~98.5% on the MNIST test set.

---

## 🌐 Demo & Deployment

* **Deployment Methods**: The application can be deployed using Gradio (for quick deployment) or Flask (for more flexibility).
* **Public Demo Link**: Once deployed on platforms like Hugging Face Spaces or Heroku, you can access the application via a public URL.

Sample screenshot of the deployed application:

![Sample UI](static/sample_prediction.png)

---

## 🔮 Future Scope

* Extend the model to recognize **letters** (EMNIST dataset) for alphanumeric recognition.
* Add **multi-digit recognition** to process forms and invoices.
* **Mobile support** for better usability across devices.
* Implement **live digit input** via webcam or touchscreen for real-time digit recognition.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
