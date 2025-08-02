#!/usr/bin/env python3
"""
FarmSmart ML Pipeline - Load Testing with Locust
Simulates realistic user behavior for performance testing
"""

from locust import HttpUser, task, between, events
import os
import random
import time
from pathlib import Path

class FarmSmartUser(HttpUser):
    """Simulates a user interacting with FarmSmart application"""
    
    wait_time = between(2, 8)  # Realistic wait time between requests
    
    def on_start(self):
        """Initialize user session"""
        print(f"FarmSmart Load Test User Started")
        
        # Find available test images
        self.test_images = self._find_test_images()
        if self.test_images:
            print(f"Found {len(self.test_images)} test images")
        else:
            print("No test images found - prediction tests will be skipped")
    
    def _find_test_images(self):
        """Find available test images in the dataset"""
        test_dirs = [
            "dataset/test",
            "data/uploads", 
            "dataset/valid"
        ]
        
        images = []
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    images.extend(Path(test_dir).glob(ext))
        
        return images[:10]  # Limit to 10 images for testing
    
    @task(5)
    def visit_homepage(self):
        """Simulate users visiting the homepage"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Homepage failed: {response.status_code}")
    
    @task(3)
    def predict_image(self):
        """Simulate image prediction requests"""
        if not self.test_images:
            # Skip if no test images available
            return
        
        # Select a random test image
        image_path = random.choice(self.test_images)
        
        try:
            with open(image_path, 'rb') as img:
                files = {'image': (image_path.name, img, 'image/jpeg')}
                
                with self.client.post("/predict", files=files, catch_response=True) as response:
                    if response.status_code == 200:
                        result = response.json()
                        if 'top_prediction' in result:
                            prediction = result['top_prediction']
                            print(f"Prediction: {prediction.get('class', 'Unknown')} "
                                  f"({prediction.get('confidence', 0):.2%})")
                            response.success()
                        else:
                            response.failure("No prediction result")
                    else:
                        response.failure(f"Prediction failed: {response.status_code}")
                        
        except Exception as e:
            print(f"Prediction error: {e}")
    
    @task(1)
    def test_retraining(self):
        """Simulate retraining requests (less frequent)"""
        # Create a simple test dataset for retraining
        test_class_name = f"test_class_{int(time.time())}"
        
        # Simulate retraining with minimal data
        data = {
            'class_name': test_class_name
        }
        
        # Create dummy files for testing
        dummy_files = []
        for i in range(3):  # Minimal test files
            dummy_files.append(('train_images', (f'train_{i}.jpg', b'dummy_image_data', 'image/jpeg')))
            dummy_files.append(('valid_images', (f'valid_{i}.jpg', b'dummy_image_data', 'image/jpeg')))
        
        try:
            with self.client.post("/custom-retrain", data=data, files=dummy_files, catch_response=True) as response:
                if response.status_code == 200:
                    result = response.json()
                    if 'message' in result:
                        print(f"Retraining initiated: {result['message']}")
                        response.success()
                    else:
                        response.failure("No retraining message")
                else:
                    response.failure(f"Retraining failed: {response.status_code}")
                    
        except Exception as e:
            print(f"Retraining error: {e}")
    
    @task(2)
    def test_static_files(self):
        """Test static file serving performance"""
        static_files = [
            "/static/logo.png",
            "/static/loader.png"
        ]
        
        file_path = random.choice(static_files)
        
        with self.client.get(file_path, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Static file failed: {response.status_code}")

class FarmSmartAdminUser(HttpUser):
    """Simulates admin user with different behavior patterns"""
    
    wait_time = between(5, 15)  # Longer wait times for admin users
    
    @task(1)
    def check_system_status(self):
        """Admin checks system status"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status check failed: {response.status_code}")

# Event listeners for better monitoring
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    print("FarmSmart Load Test Starting...")
    print(f"Target: {environment.host}")
    print(f"Users: {environment.runner.user_count if environment.runner else 'N/A'}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    print("FarmSmart Load Test Completed!")

# Configuration for different test scenarios
class QuickTest(FarmSmartUser):
    """Quick test with minimal load"""
    wait_time = between(1, 3)
    
    @task(1)
    def quick_prediction(self):
        """Quick prediction test"""
        self.predict_image()

class StressTest(FarmSmartUser):
    """Stress test with high load"""
    wait_time = between(0.5, 2)
    
    @task(10)
    def stress_prediction(self):
        """High-frequency prediction requests"""
        self.predict_image()

# Usage Instructions:
"""
To run load tests:

1. Start the FarmSmart application:
   python src/app.py

2. Run basic load test:
   locust -f src/locustfile.py --host=http://localhost:5000

3. Run stress test:
   locust -f src/locustfile.py --host=http://localhost:5000 --users=50 --spawn-rate=5

4. Run quick test:
   locust -f src/locustfile.py --host=http://localhost:5000 --users=10 --spawn-rate=2

5. Open browser and go to http://localhost:8089 to view results

Test Scenarios:
- Homepage visits (most common)
- Image predictions (core functionality)
- Static file serving (performance)
- Retraining requests (less frequent)
- Admin monitoring (different user type)
"""
