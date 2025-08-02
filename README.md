# FarmSmart Disease Classifier

FarmSmart is a machine learning pipeline that helps identify plant diseases from leaf images using deep learning. It supports prediction, model retraining with new data, and a web UI for seamless interaction.


## Features
- ✅ CNN model using Keras with regularization, callbacks, and transfer learning
- ✅ Preprocessing with advanced augmentation using `ImageDataGenerator`
- ✅ RESTful API with Flask for prediction and retraining
- ✅ Retrain pipeline using uploaded ZIP files
- ✅ User Interface (Flask + HTML) with Tailwind styling
- ✅ Model Evaluation notebook with Accuracy, Loss, Precision, Recall, F1
- ✅ Load testing with Locust to simulate concurrent requests


## Project Structure

FarmSmart/
├── app.py                     # Flask application with UI and API endpoints
├── locustfile.py              # Load testing file
├── frontend/
│   └── index.html             # Web UI template (uses Tailwind CSS)
├── models/
│   └── farmsmart_diseases.keras # Trained model
├── data/
│   ├── train/                 # Training images
│   ├── valid/                 # Validation images
│   └── uploads/               # Uploaded images for prediction
├── notebook/
│   └── farmsmart_pipeline.ipynb # Model training and evaluation notebook
├── src/
│   ├── preprocessing.py       # Image generators and augmentations
│   ├── model.py               # CNN architecture and training logic
│   ├── prediction.py          # Model prediction logic
│   ├── retrain.py             # Model retraining logic
│   └── train_model.py         # Training entrypoint
└── README.md
```


## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/FarmSmart.git
cd FarmSmart

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Train model (optional)
python -m src.train_model

# Run the app
python app.py
```

---

## Prediction
Send an image file to `/predict` via UI or API.

```
POST /predict
Form-Data:
- image: <your_image.jpg>
```

## Retraining
Upload a ZIP file with new train/valid images to trigger retraining.

```
POST /retrain
Form-Data:
- dataset: <your_dataset.zip>
```



## Model Evaluation Metrics

| Metric        | Result (sample) |
|---------------|------------------|
| Accuracy      | ✅ 0.85+         |
| Precision     | ✅ 0.84+         |
| Recall        | ✅ 0.83+         |
| F1 Score      | ✅ 0.835+        |
| Top-3 Accuracy| ✅ 0.95+         |

> Detailed plots and results in `notebook/farmsmart_pipeline.ipynb`

---

## Locust Load Testing

```bash
locust -f locustfile.py
# Visit http://localhost:8089 to simulate load
```

Tested with 1, 10, 50 users — monitored response times and latency.

---

## Video Demo
📺 [YouTube Link](https://your-link.com)

---

## Deployment (Optional)
- Hosted on: [Heroku / Render / Azure / etc.]
- Live URL: `https://farmsmart-demo.com`

---

## Contributors
- **Your Name** – Model & API Engineer
- **Collaborators** – Frontend & Load Testing

---
## License
MIT License © 2025 African Leadership University
