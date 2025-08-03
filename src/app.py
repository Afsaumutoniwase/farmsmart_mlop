# === Simplified Flask App ===
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import sqlite3
from prediction import predict_image
from retrain import *

app = Flask(__name__, template_folder="template", static_folder="static")

UPLOAD_FOLDER = "dataset/uploads"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "farmsmart.keras")

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

# Database initialization
def init_db():
    conn = sqlite3.connect('farmsmart.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_name TEXT NOT NULL,
                  predicted_class TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS retraining_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  dataset_name TEXT NOT NULL,
                  start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  end_time TIMESTAMP,
                  status TEXT NOT NULL,
                  accuracy REAL,
                  new_samples INTEGER,
                  training_duration REAL)''')
    conn.commit()
    conn.close()

init_db()

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

    try:
        results = predict_image(MODEL_PATH, CLASS_NAMES, image_path)
        top_result = results[0] if results else {}
        print(f" Raw prediction results: {results}")


        if top_result:
            conn = sqlite3.connect('farmsmart.db')
            c = conn.cursor()
            c.execute('''INSERT INTO predictions (image_name, predicted_class, confidence)
                         VALUES (?, ?, ?)''',
                      (filename, top_result['class'], top_result['confidence']))
            conn.commit()
            conn.close()

        return jsonify({"top_prediction": top_result, "all_predictions": results})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/custom-retrain", methods=["POST"])
def custom_retrain():
    retrain_mode = request.form.get("retrain_mode")
    class_name = request.form.get("class_name")
    train_files = request.files.getlist("train_images")
    valid_files = request.files.getlist("valid_images")

    conn = sqlite3.connect('../farmsmart.db')
    c = conn.cursor()
    retrain_id = None

    try:
        if retrain_mode == "existing":
            dataset_name = "existing_retrain_dataset"
            c.execute('''INSERT INTO retraining_history (dataset_name, status, new_samples) VALUES (?, ?, ?)''',
                      (dataset_name, 'started', 0))
            retrain_id = c.lastrowid
            conn.commit()

            message = retrain_model_with_fixed_data()

        elif retrain_mode == "new":
            if not class_name or not train_files or not valid_files:
                return jsonify({"error": "Missing input for custom class retraining"}), 400

            train_path = os.path.join("../dataset/retrain/train", class_name)
            valid_path = os.path.join("../dataset/retrain/valid", class_name)
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(valid_path, exist_ok=True)

            for file in train_files:
                if file and file.filename:
                    path = os.path.join(train_path, secure_filename(file.filename))
                    file.save(path)

            for file in valid_files:
                if file and file.filename:
                    path = os.path.join(valid_path, secure_filename(file.filename))
                    file.save(path)

            c.execute('''INSERT INTO retraining_history (dataset_name, status, new_samples) VALUES (?, ?, ?)''',
                      (class_name, 'started', len(train_files) + len(valid_files)))
            retrain_id = c.lastrowid
            conn.commit()

            message = retrain_model_with_fixed_data()

        else:
            return jsonify({"error": "Invalid retrain mode"}), 400

        c.execute('''UPDATE retraining_history 
                     SET status = ?, end_time = CURRENT_TIMESTAMP, accuracy = ?
                     WHERE id = ?''', ('completed', 0.85, retrain_id))
        conn.commit()
        return jsonify({"message": message})

    except Exception as e:
        if retrain_id:
            c.execute('''UPDATE retraining_history 
                         SET status = ?, end_time = CURRENT_TIMESTAMP
                         WHERE id = ?''', ('failed', retrain_id))
            conn.commit()
        return jsonify({"error": f"Retraining failed: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
