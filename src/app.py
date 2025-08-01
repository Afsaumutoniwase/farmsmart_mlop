# === app.py (Flask Frontend UI with Tailwind) ===
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from prediction import predict_image
from retrain import retrain_model

app = Flask(__name__, template_folder="template", static_folder="static")

UPLOAD_FOLDER = "data/uploads"
RETRAIN_FOLDER = "retrain_uploads"
MODEL_PATH = "models/farmsmart.keras"

CLASS_NAMES = [
    'Pepper_bell_Bacterial_spot',
    'Pepper_bell_healthy',
    'Strawberry_healthy',
    'Strawberry_Leaf_scorch',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Target_Spot'
]


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RETRAIN_FOLDER'] = RETRAIN_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    image_file.save(image_path)

    results = predict_image(MODEL_PATH, CLASS_NAMES, image_path)
    top_result = results[0] if results else {}
    return jsonify({"top_prediction": top_result, "all_predictions": results})

@app.route("/custom-retrain", methods=["POST"])
def custom_retrain():
    class_name = request.form.get("class_name")
    train_files = request.files.getlist("train_images")
    valid_files = request.files.getlist("valid_images")

    if not class_name or not train_files or not valid_files:
        return jsonify({"error": "Missing input"}), 400

    train_path = os.path.join("data/custom/train", class_name)
    valid_path = os.path.join("data/custom/valid", class_name)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)

    for file in train_files:
        file.save(os.path.join(train_path, secure_filename(file.filename)))
    for file in valid_files:
        file.save(os.path.join(valid_path, secure_filename(file.filename)))

    message = retrain_model("data/custom/train", "data/custom/valid")
    return jsonify({"message": message})

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
