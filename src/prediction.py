from pyexpat import model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def predict_image(model_path, class_names, image_path, img_size=(128, 128), top_k=3):
    model = load_model(model_path)
    model_TEST = tf.keras.models.load_model("../models/farmsmart.keras")
    print("🧪 Model output shape:", model_TEST.output_shape)


    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]

    if len(predictions) != len(class_names):
        return [{
            "class": "Class mismatch error",
            "confidence": 0.0,
            "percent": 0.0
        }]

    if np.isnan(predictions).any():
        return [{
            "class": "Invalid prediction (NaN)",
            "confidence": 0.0,
            "percent": 0.0
        }]

    # Get top_k results
    top_indices = predictions.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "class": class_names[idx],
            "confidence": float(predictions[idx]),
            "percent": float(predictions[idx] * 100)
        })

    return results
