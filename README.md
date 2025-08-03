
# FarmSmart Disease Classifier

**African Leadership University**  
**Bachelor of Software Engineering (BSE)**  
**Machine Learning Pipeline - Summative Assignment** by 
**Afsa Umutoniwase**

---

## Project Overview

FarmSmart Disease Classifier is an end-to-end machine learning pipeline designed to detect plant diseases from images. Built using TensorFlow and Flask, the solution supports:

- Image classification (plant disease prediction)
- Model evaluation with advanced metrics
- Custom retraining triggered via UI or API
- Real-time API deployment with Locust load testing
- Clear visualization of training and evaluation metrics
- Full-featured Web UI for non-technical users

This pipeline demonstrates the **complete ML lifecycle** from data ingestion to deployment on a simulated production server.

---

## Objective

To build, evaluate, and deploy a machine learning classifier on **image data**, incorporating:

- A prediction API
- A retraining trigger
- Evaluation metrics (accuracy, loss, precision, recall, AUC, F1)
- UI + Deployment setup
- Load testing via Locust
- Upload/retrain functionality

---

## Project Structure

```
FarmSmart/
│
├── README.md
├── notebook/
│   └── farmsmart_notebook.ipynb         ← Full ML pipeline notebook
│
├── src/
│   ├── app.py                           ← Flask app (API endpoints)
│   ├── model.py                         ← Model training + evaluation
│   ├── preprocessing.py                 ← Data loading utilities
│   ├── prediction.py                    ← Image prediction logic
│   └── locustfile.py                    ← Load testing script
│   └── static/
│       └── logo.png                     ← Web UI assets
│
│   └──templates/
│      └── index.html                    ← HTML UI
│
├── dataset/
│   ├── train/                           ← Training images
│   ├── valid/                           ← Validation images
│   ├── test/                            ← Single image test set
│   └── retrain/                         ← Folder for retrain uploads
│       ├── train/
│       └── valid/
│
├── models/
│   └── farmsmart.keras                  ← Saved model file
│
│
├── farmsmart.db                         ← SQLite DB for predictions & retrain logs
```

---

## Setup Instructions

### Requirements

- Python 3.8+
- TensorFlow
- Flask
- Locust
- matplotlib, seaborn, scikit-learn, etc.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the App

### 1. Train Model (Offline)

```bash
python src/model.py
```

This script builds, trains, evaluates, and saves the model (`models/farmsmart.keras`).

### 2. Start Flask API

```bash
cd src
python app.py
```

- Navigate to `http://localhost:5000`

---

## Using the Web UI

### Access the Dashboard

Go to `http://localhost:5000` in your browser.

### How to Use

1. **Predict Disease**  
   - Upload a plant image in JPG/PNG format
   - Click "Predict Disease"
   - View predicted class and confidence

2. **Retrain Model**  
   - Fill in a new class name (e.g., `Tomato_Blight`)
   - Upload at least **10 training** and **5 validation** images
   - Click "Retrain Model"
   - View status after retraining is complete

3. **System Info + Class Summary**  
   - Shown on the right column in UI
   - Includes supported image types, class count, and model type

> ![UI Screenshot](![alt text](image.png))

---

## Load Testing with Locust

```bash
locust -f src/locustfile.py --host=http://localhost:5000
```

Go to `http://localhost:8089` and simulate user traffic:
- Predict requests
- Retrain uploads
- Homepage visits

---

## Features Summary

### Prediction
- Single image upload
- Class label & confidence score

### Retraining
- New class retraining via UI or API
- Image count checks (5+ validation, 10+ training)
- Accuracy recorded to database

### Evaluation Metrics
- Accuracy, Loss, Precision, Recall
- AUC, F1, Cohen Kappa, MCC
- Confusion matrix & bar plots

### Web UI
- Intuitive upload interface
- Instant visual feedback
- Icons and error handling

### Load Testing
- Scalable Locust test simulation
- Visual latency/throughput tracking

---

## Video Demo

[YouTube Demo Link](https://your-demo-link.com)  
_(Includes both prediction and retraining demonstration)_

---

## Results from Load Simulation

- **Users simulated:** 7
- **Endpoints tested:** `/`, `/predict`, `/custom-retrain`
- **Response time:** avg 2600ms (predict), retrain varies
- **Errors handled:** gracefully skipped if data invalid
> ![UI Screenshot](![alt text](image.png))
---

## Conclusion

This project meets all the summative requirements including UI, API, retraining, prediction, cloud deployability, and performance simulation.

It is a complete ML solution built for scale and usability.
