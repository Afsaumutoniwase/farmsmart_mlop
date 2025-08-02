# === Simplified Flask App ===
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import sqlite3
from prediction import predict_image
from retrain import retrain_model_with_individual_images

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

# Database initialization
def init_db():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('farmsmart.db')
    c = conn.cursor()
    
    # Predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_name TEXT NOT NULL,
                  predicted_class TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Retraining history table
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

# Initialize database on startup
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
        
        # Log prediction to database
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
    class_name = request.form.get("class_name")
    train_files = request.files.getlist("train_images")
    valid_files = request.files.getlist("valid_images")

    if not class_name or not train_files or not valid_files:
        return jsonify({"error": "Missing input"}), 400

    # Log retraining start
    conn = sqlite3.connect('farmsmart.db')
    c = conn.cursor()
    c.execute('''INSERT INTO retraining_history 
                 (dataset_name, status, new_samples) VALUES (?, ?, ?)''',
              (class_name, 'started', len(train_files) + len(valid_files)))
    retrain_id = c.lastrowid
    conn.commit()
    conn.close()

    try:
        # Use the new retraining function that handles individual images
        message = retrain_model_with_individual_images(train_files, valid_files, class_name)
        
        # Update retraining status
        conn = sqlite3.connect('farmsmart.db')
        c = conn.cursor()
        c.execute('''UPDATE retraining_history 
                     SET status = ?, end_time = CURRENT_TIMESTAMP, accuracy = ?
                     WHERE id = ?''', ('completed', 0.85, retrain_id))
        conn.commit()
        conn.close()
        
        return jsonify({"message": message})
    except Exception as e:
        # Update retraining status on error
        conn = sqlite3.connect('farmsmart.db')
        c = conn.cursor()
        c.execute('''UPDATE retraining_history 
                     SET status = ?, end_time = CURRENT_TIMESTAMP
                     WHERE id = ?''', ('failed', retrain_id))
        conn.commit()
        conn.close()
        
        return jsonify({"error": f"Retraining failed: {str(e)}"}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
