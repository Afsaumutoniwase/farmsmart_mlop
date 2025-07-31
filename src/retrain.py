

# === Enhanced Retraining Module with Separate Training/Validation ===
import os
import zipfile
import tarfile
import shutil
import tempfile
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def extract_archive(archive_path, extract_to):
    """Extract ZIP, TAR, or GZ files"""
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar') or archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

def validate_dataset_structure(data_path):
    """Validate that the dataset has proper class folder structure"""
    if not os.path.exists(data_path):
        return False, "Data path does not exist"
    
    # Check for class folders
    class_folders = [d for d in os.listdir(data_path) 
                    if os.path.isdir(os.path.join(data_path, d))]
    
    if len(class_folders) == 0:
        return False, "No class folders found"
    
    # Check each class folder for images
    total_images = 0
    for class_folder in class_folders:
        class_path = os.path.join(data_path, class_folder)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        total_images += len(images)
        
        if len(images) < 10:  # Minimum 10 images per class
            return False, f"Class '{class_folder}' has only {len(images)} images (minimum 10 required)"
    
    if total_images < 50:  # Minimum 50 total images
        return False, f"Total images ({total_images}) is below minimum requirement (50)"
    
    return True, f"Valid dataset with {len(class_folders)} classes and {total_images} images"

def retrain_model_with_separate_data(train_path, val_path):
    """Retrain model with separate training and validation datasets"""
    
    print(f"🔄 Starting retraining with separate datasets...")
    print(f"  Training data: {train_path}")
    print(f"  Validation data: {val_path}")
    
    # Create temporary directories for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        train_extract = os.path.join(temp_dir, "train")
        val_extract = os.path.join(temp_dir, "val")
        os.makedirs(train_extract, exist_ok=True)
        os.makedirs(val_extract, exist_ok=True)
        
        # Extract training data
        print("📦 Extracting training data...")
        extract_archive(train_path, train_extract)
        
        # Extract validation data
        print("📦 Extracting validation data...")
        extract_archive(val_path, val_extract)
        
        # Validate both datasets
        print("🔍 Validating training dataset...")
        train_valid, train_msg = validate_dataset_structure(train_extract)
        if not train_valid:
            raise ValueError(f"Training dataset validation failed: {train_msg}")
        
        print("🔍 Validating validation dataset...")
        val_valid, val_msg = validate_dataset_structure(val_extract)
        if not val_valid:
            raise ValueError(f"Validation dataset validation failed: {val_msg}")
        
        # Check if both datasets have same classes
        train_classes = set(os.listdir(train_extract))
        val_classes = set(os.listdir(val_extract))
        
        if train_classes != val_classes:
            missing_in_val = train_classes - val_classes
            missing_in_train = val_classes - train_classes
            raise ValueError(f"Class mismatch: Training has {train_classes}, Validation has {val_classes}")
        
        print(f"✅ Both datasets validated successfully!")
        print(f"  Classes: {sorted(train_classes)}")
        
        # Load existing model
        model_path = "models/farmsmart_diseases.keras"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print("🤖 Loading existing model...")
        model = tf.keras.models.load_model(model_path)
        
        # Prepare data generators
        print("📊 Setting up data generators...")
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_extract,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_extract,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"📈 Training samples: {train_generator.samples}")
        print(f"📈 Validation samples: {val_generator.samples}")
        
        # Compile model for retraining
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks for retraining
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/farmsmart_diseases_retrained.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Retrain the model
        print("🚀 Starting model retraining...")
        history = model.fit(
            train_generator,
            epochs=10,  # Fewer epochs for retraining
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the retrained model
        print("📊 Evaluating retrained model...")
        val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_generator, verbose=0)
        
        # Save the retrained model
        final_model_path = 'models/farmsmart_diseases_retrained.keras'
        model.save(final_model_path)
        
        # Create backup of original model
        backup_path = f'models/farmsmart_diseases_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'
        if os.path.exists(model_path):
            shutil.copy2(model_path, backup_path)
        
        # Replace original model with retrained version
        shutil.move(final_model_path, model_path)
        
        print(f"✅ Retraining completed successfully!")
        print(f"  Final validation accuracy: {val_accuracy:.4f}")
        print(f"  Final validation precision: {val_precision:.4f}")
        print(f"  Final validation recall: {val_recall:.4f}")
        print(f"  Model saved to: {model_path}")
        print(f"  Original model backed up to: {backup_path}")
        
        return f"Model retrained successfully! Validation accuracy: {val_accuracy:.2%}"

def retrain_model(zip_path):
    """Legacy function for backward compatibility"""
    return retrain_model_with_separate_data(zip_path, zip_path)
