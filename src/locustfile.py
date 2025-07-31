# locustfile.py
from locust import HttpUser, task, between
import os

class FarmSmartUser(HttpUser):
    wait_time = between(1, 5)  # Simulate user wait time between requests

    @task(3)
    def predict_image(self):
        image_path = "sample.jpg"  # Replace with a real image
        if os.path.exists(image_path):
            with open(image_path, 'rb') as img:
                files = {'image': (image_path, img, 'image/jpeg')}
                self.client.post("/predict", files=files)
        else:
            print("⚠️ sample.jpg not found")

    @task(1)
    def retrain_model(self):
        dataset_path = "sample_dataset.zip"  # Replace with a real zip file
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as zip_file:
                files = {'dataset': (dataset_path, zip_file, 'application/zip')}
                self.client.post("/retrain", files=files)
        else:
            print("⚠️ sample_dataset.zip not found")
