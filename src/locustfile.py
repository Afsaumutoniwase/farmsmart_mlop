# src/locustfile.py

from locust import HttpUser, task, between
import os
import random
from pathlib import Path

class FarmSmartUser(HttpUser):
    wait_time = between(2, 5)

    def on_start(self):
        self.train_dir = "../dataset/retrain/train"
        self.valid_dir = "../dataset/retrain/valid"
        self.train_class_map = self._group_images_by_class(self.train_dir)
        self.valid_class_map = self._group_images_by_class(self.valid_dir)
        self.common_classes = list(set(self.train_class_map) & set(self.valid_class_map))
        self.all_test_images = [img for imgs in self.train_class_map.values() for img in imgs]

    def _group_images_by_class(self, directory):
        class_map = {}
        root = Path(directory)
        if not root.exists():
            return class_map

        for class_dir in root.iterdir():
            if class_dir.is_dir():
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    images.extend(class_dir.glob(ext))
                if images:
                    class_map[class_dir.name] = images
        return class_map

    @task(1)
    def visit_homepage(self):
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Homepage failed: {response.status_code}")

    @task(3)
    def predict_image(self):
        if not self.all_test_images:
            return

        image_path = random.choice(self.all_test_images)
        mime_type = 'image/png' if image_path.suffix.lower() == '.png' else 'image/jpeg'

        with open(image_path, 'rb') as img:
            files = {'image': (image_path.name, img, mime_type)}
            with self.client.post("/predict", files=files, catch_response=True) as response:
                try:
                    result = response.json()
                    if response.status_code == 200 and "top_prediction" in result:
                        response.success()
                    else:
                        response.failure("Prediction failed or missing result")
                except Exception as e:
                    response.failure(f"Invalid JSON response: {e}")

    @task(2)
    def retrain_model(self):
        if not self.common_classes:
            print("No common classes in train/valid")
            return

        selected_class = random.choice(self.common_classes)

        data = {
            "class_name": selected_class,
            "retrain_mode": "existing"
        }

        with self.client.post("/custom-retrain", data=data, catch_response=True) as response:
            try:
                result = response.json()
                if response.status_code == 200 and "message" in result:
                    response.success()
                else:
                    response.failure(f"Retraining failed or incomplete: {result}")
            except Exception as e:
                response.failure(f"Retrain JSON error: {e}")
