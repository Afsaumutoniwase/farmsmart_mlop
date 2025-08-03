# src/locustfile.py

from locust import HttpUser, task, between
import os
import random
from pathlib import Path

class FarmSmartUser(HttpUser):
    wait_time = between(2, 5)

    def on_start(self):
        """Load test images from dataset/test and dataset/valid."""
        
        self.test_images = self._find_images("../dataset/retrain/train")
        self.valid_images = self._find_images("../dataset/retrain/valid")


    def _find_images(self, directory):
        if not os.path.exists(directory):
            return []

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(Path(directory).rglob(ext))  # Use rglob instead of glob
        return list(images)
    
    def _group_images_by_class(self, directory):
        """Group images by class folder (e.g. retrain/train/class_x/*.jpg)."""
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
        """Simulate visiting the homepage."""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Homepage failed: {response.status_code}")

    @task(3)
    def predict_image(self):
        """Simulate image prediction (1 image only)"""
        if not self.test_images:
            return

        image_path = random.choice(self.test_images)
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
        """Simulate retraining using 5+ train and 5+ valid images from a single class."""
        train_class_map = self._group_images_by_class("../dataset/retrain/train")
        valid_class_map = self._group_images_by_class("../dataset/retrain/valid")

        common_classes = set(train_class_map) & set(valid_class_map)
        if not common_classes:
            print("No common class folders in train and valid.")
            return

        selected_class = random.choice(list(common_classes))
        train_imgs = train_class_map[selected_class]
        valid_imgs = valid_class_map[selected_class]

        if len(train_imgs) < 5 or len(valid_imgs) < 5:
            print(f"Class '{selected_class}' has insufficient images.")
            return

        class_name = f"{selected_class}_{random.randint(1000, 9999)}"
        data = {"class_name": class_name}

        files = []
        for i in range(5):
            with open(train_imgs[i], "rb") as t_img, open(valid_imgs[i], "rb") as v_img:
                train_filename = f"{selected_class}_train_{i}.jpg"
                valid_filename = f"{selected_class}_valid_{i}.jpg"
                files.append(("train_images", (train_filename, t_img.read(), "image/jpeg")))
                files.append(("valid_images", (valid_filename, v_img.read(), "image/jpeg")))

        with self.client.post("/custom-retrain", data=data, files=files, catch_response=True) as response:
            try:
                result = response.json()
                print("Retrain response:", result)
                if response.status_code == 200 and "message" in result:
                    response.success()
                else:
                    response.failure(f"Retraining failed or incomplete: {result}")
            except Exception as e:
                response.failure(f"Retrain JSON error: {e}")
