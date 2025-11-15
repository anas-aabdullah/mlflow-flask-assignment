import os
import requests
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# =========================
# Configuration
# =========================
MODEL_LOCAL_PATH = "model.h5"
DAGSHUB_ARTIFACT_URL = "https://dagshub.com/anas-aabdullah/mlflow-flask-assignment/raw/branch/main/mlruns/0/f6977b8356bb4debb4e98812a1bc276a/artifacts/model.h5"
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# =========================
# Download model if not exists
# =========================
if not os.path.exists(MODEL_LOCAL_PATH):
    print("Model not found locally. Downloading from DagsHub...")
    r = requests.get(DAGSHUB_ARTIFACT_URL, allow_redirects=True)
    with open(MODEL_LOCAL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded successfully!")

# =========================
# Load model
# =========================
print("Loading model...")
model = load_model(MODEL_LOCAL_PATH)
print("Model loaded successfully!")

# =========================
# Helper: Predict function
# =========================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(180, 180))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    return {"class": CLASS_NAMES[class_idx], "confidence": float(preds[0][class_idx])}

# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)
        result = predict_image(file_path)
        return render_template("upload.html", result=result)
    return render_template("upload.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)
    result = predict_image(file_path)
    return jsonify(result)

# =========================
# Run Flask
# =========================
if __name__ == "__main__":
    app.run(debug=True)
