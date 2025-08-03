# === Retraining Module with Fixed Dataset Paths ===
import os
import shutil
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
# Add at the top of retrain.py
import threading
RETRAIN_LOCK = threading.Lock()

# Paths
TRAIN_DIR = "../dataset/retrain/train"
VALID_DIR = "../dataset/retrain/valid"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "farmsmart.keras")

def validate_dataset_structure(data_path):
    if not os.path.exists(data_path):
        return False, "Data path does not exist"

    class_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if len(class_folders) == 0:
        return False, "No class folders found"

    total_images = 0
    for class_folder in class_folders:
        class_path = os.path.join(data_path, class_folder)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        total_images += len(images)

        if len(images) < 5:
            return False, f"Class '{class_folder}' has only {len(images)} images (minimum 5 required)"

    if total_images < 15:
        return False, f"Total images ({total_images}) is below minimum requirement (15)"

    return True, f"Valid dataset with {len(class_folders)} classes and {total_images} images"

def retrain_model_with_fixed_data():
    with RETRAIN_LOCK:

        print("Starting retraining using fixed folders")
        try:
            # Validate dataset structure
            for path, name in [(TRAIN_DIR, "Training"), (VALID_DIR, "Validation")]:
                is_valid, msg = validate_dataset_structure(path)
                if not is_valid:
                    raise ValueError(f"{name} dataset invalid: {msg}")

            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

            print("Loading original model...")
            original_model = tf.keras.models.load_model(MODEL_PATH)

            print("Setting up dummy input...")
            dummy_input = tf.keras.Input(shape=(128, 128, 3))

            try:
                _ = original_model(dummy_input)
            except Exception:
                original_model.build(input_shape=(None, 128, 128, 3))

            print("Cloning model for retraining...")
            cloned_model = tf.keras.models.clone_model(original_model)
            cloned_model.build(input_shape=(None, 128, 128, 3))
            cloned_model.set_weights(original_model.get_weights())

            print("Setting up data generators...")
            train_gen = ImageDataGenerator(
                rescale=1. / 255, rotation_range=20, width_shift_range=0.2,
                height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                horizontal_flip=True, fill_mode='nearest')
            val_gen = ImageDataGenerator(rescale=1. / 255)

            train_data = train_gen.flow_from_directory(
                TRAIN_DIR, target_size=(128, 128), batch_size=32,
                class_mode='categorical', shuffle=True)
            val_data = val_gen.flow_from_directory(
                VALID_DIR, target_size=(128, 128), batch_size=32,
                class_mode='categorical', shuffle=False)

            num_classes = train_data.num_classes
            print(f"Detected {num_classes} output classes")

            print("Preparing new final layer...")
            if isinstance(cloned_model, tf.keras.Sequential):
                model = tf.keras.Sequential(name="retrained_sequential_model")
                for layer in cloned_model.layers[:-1]:
                    model.add(layer)
                model.add(tf.keras.layers.Dense(num_classes, activation="softmax", name=f"output_dense_{num_classes}"))
            else:
                base_output = cloned_model.layers[-2].output
                new_output = tf.keras.layers.Dense(num_classes, activation='softmax', name=f"output_dense_{num_classes}")(base_output)
                model = tf.keras.Model(inputs=cloned_model.input, outputs=new_output, name="retrained_model")

            print("Compiling model...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )

            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(MODEL_DIR, 'farmsmart_retrained.keras'),
                    monitor='val_accuracy', save_best_only=True, verbose=1)
            ]

            print("Training...")
            model.fit(train_data, epochs=10, validation_data=val_data, callbacks=callbacks, verbose=1)

            print("Evaluating...")
            val_loss, val_acc, val_prec, val_recall = model.evaluate(val_data, verbose=0)
            model.save(os.path.join(MODEL_DIR, 'farmsmart_retrained.keras'))


            print(f"Retraining completed successfully! Accuracy: {val_acc:.4f}")
            return f"Model retrained. Val Accuracy: {val_acc:.2%}"

        except Exception as e:
            print(f"Retraining failed: {str(e)}")
            raise e
