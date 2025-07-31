# === src/api.py ===
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from .prediction import predict_image
from .retrain import retrain_model

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "data/uploads"
RETRAIN_FOLDER = "data/retrain_uploads"
MODEL_PATH = "models/farmsmart_diseases.keras"
CLASS_NAMES = ['Cassava Mosaic', 'Early Blight', 'Late Blight', 'Rust', 'Healthy', 'Scab', 'Mildew']  # Replace with real class names

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RETRAIN_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return "🌿 FarmSmart Disease Classifier API is running."

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    results = predict_image(MODEL_PATH, CLASS_NAMES, image_path)
    return jsonify(results)

@app.route("/retrain", methods=["POST"])
def retrain():
    if 'dataset' not in request.files:
        return jsonify({"error": "No dataset uploaded"}), 400

    dataset = request.files['dataset']
    filename = secure_filename(dataset.filename)
    zip_path = os.path.join(RETRAIN_FOLDER, filename)
    dataset.save(zip_path)

    success, message = retrain_model(zip_path)
    if not success:
        return jsonify({"error": message}), 500

    return jsonify({"message": message})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
