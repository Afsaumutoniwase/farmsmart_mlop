

# === Enhanced Retraining Module with Individual Image Upload ===
import os
import zipfile
import tarfile
import shutil
import tempfile
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from werkzeug.utils import secure_filename

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
        
        if len(images) < 5:  # Minimum 5 images per class
            return False, f"Class '{class_folder}' has only {len(images)} images (minimum 5 required)"
    
    if total_images < 15:  # Minimum 15 total images
        return False, f"Total images ({total_images}) is below minimum requirement (15)"
    
    return True, f"Valid dataset with {len(class_folders)} classes and {total_images} images"

def organize_uploaded_images(train_files, valid_files, class_name):
    """Organize uploaded individual images into proper folder structure"""
    print(f"Organizing uploaded images for class: {class_name}")
    
    # Create directories
    train_dir = os.path.join("data/custom/train", class_name)
    valid_dir = os.path.join("data/custom/valid", class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    
    # Save training images
    train_count = 0
    for file in train_files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(train_dir, filename)
            file.save(file_path)
            train_count += 1
    
    # Save validation images
    valid_count = 0
    for file in valid_files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(valid_dir, filename)
            file.save(file_path)
            valid_count += 1
    
    print(f"Organized {train_count} training images and {valid_count} validation images")
    return train_dir, valid_dir

def retrain_model_with_individual_images(train_files, valid_files, class_name):
    """Retrain model with individually uploaded images"""
    
    print(f"Starting retraining with individual images for class: {class_name}")
    
    try:
        # Organize uploaded images into folders
        train_dir, valid_dir = organize_uploaded_images(train_files, valid_files, class_name)
        
        # Validate the organized dataset
        print("Validating organized dataset...")
        train_valid, train_msg = validate_dataset_structure("data/custom/train")
        if not train_valid:
            raise ValueError(f"Training dataset validation failed: {train_msg}")
        
        valid_valid, valid_msg = validate_dataset_structure("data/custom/valid")
        if not valid_valid:
            raise ValueError(f"Validation dataset validation failed: {valid_msg}")
        
        print("Dataset validation successful!")
        
        # Load existing model
        model_path = "models/farmsmart.keras"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_path)
        
        # Prepare data generators
        print("Setting up data generators...")
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
            "data/custom/train",
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            "data/custom/valid",
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        
        # Compile model for retraining
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
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
                filepath='models/farmsmart_retrained.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Retrain the model
        print("Starting model retraining...")
        history = model.fit(
            train_generator,
            epochs=10,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the retrained model
        print("Evaluating retrained model...")
        val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_generator, verbose=0)
        
        # Save the retrained model
        final_model_path = 'models/farmsmart_retrained.keras'
        model.save(final_model_path)
        
        # Create backup of original model
        backup_path = f'models/farmsmart_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'
        if os.path.exists(model_path):
            shutil.copy2(model_path, backup_path)
        
        # Replace original model with retrained version
        shutil.move(final_model_path, model_path)
        
        print(f"Retraining completed successfully!")
        print(f"  Final validation accuracy: {val_accuracy:.4f}")
        print(f"  Final validation precision: {val_precision:.4f}")
        print(f"  Final validation recall: {val_recall:.4f}")
        print(f"  Model saved to: {model_path}")
        print(f"  Original model backed up to: {backup_path}")
        
        return f"Model retrained successfully! Validation accuracy: {val_accuracy:.2%}"
        
    except Exception as e:
        print(f"Retraining failed: {str(e)}")
        raise e

def retrain_model(zip_path):
    """Legacy function for backward compatibility"""
    return retrain_model_with_individual_images(zip_path, zip_path, "default_class")
