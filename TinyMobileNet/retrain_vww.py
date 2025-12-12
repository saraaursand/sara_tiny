import os
import datetime
import numpy as np
from absl import app
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import glob
import csv
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ==================================================================================
# CONFIGURATION AND VALIDATION
# ==================================================================================

def load_config(config_path='retrain_config.yaml'):
    """Load and validate configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if not config:
            raise ValueError(f"Config file '{config_path}' is empty or invalid.")
    
    return config

def validate_config(config):
    """Validate that all required config parameters are present."""
    required_keys = [
        'MODEL_PATH', 'IMAGE_SIZE', 'BATCH_SIZE', 'EPOCHS', 
        'BASE_DIR', 'LEARNING_RATE', 'TEST_IMAGES_DIR', 'TEST_OUTPUT_DIR'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config parameters: {', '.join(missing_keys)}")
    
    return True

def validate_paths(config):
    """Validate that required paths exist."""
    if not os.path.exists(config['MODEL_PATH']):
        raise FileNotFoundError(f"Model path does not exist: {config['MODEL_PATH']}")
    
    if not os.path.exists(config['BASE_DIR']):
        raise FileNotFoundError(f"Base directory does not exist: {config['BASE_DIR']}")
    
    if not os.path.exists(config['TEST_IMAGES_DIR']):
        raise FileNotFoundError(f"Test images directory does not exist: {config['TEST_IMAGES_DIR']}")
    
    return True


# ==================================================================================
# IMAGE PROCESSING
# ==================================================================================

def get_image_files(directory):
    """Get all image files from directory recursively."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    
    return sorted(image_files)

def get_labels_from_structure(test_dir):
    """Extract labels from directory structure (class_name/images)."""
    labels_dict = {}
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                class_label = os.path.basename(root)
                labels_dict[file_path] = class_label
    
    return labels_dict

def load_and_preprocess_image(image_path, image_size):
    """Load and preprocess image with rescale matching training."""
    img = load_img(image_path, target_size=(image_size, image_size), color_mode='rgb')
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array

def prepare_test_dataset(test_images_dir, image_size):
    """Prepare test dataset from directory structure."""
    image_files = get_image_files(test_images_dir)
    
    if not image_files:
        raise ValueError(f"No images found in {test_images_dir}")
    
    labels_dict = get_labels_from_structure(test_images_dir)
    unique_classes = sorted(set(labels_dict.values()))
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    images = []
    true_labels = []
    
    for img_path in image_files:
        try:
            img_array = load_and_preprocess_image(img_path, image_size)
            images.append(img_array)
            true_labels.append(class_to_idx[labels_dict[img_path]])
        except Exception:
            continue
    
    images = np.array(images)
    true_labels = np.array(true_labels)
    
    return images, true_labels, image_files


# ==================================================================================
# MODEL EVALUATION
# ==================================================================================

def evaluate_model(model, test_images, test_labels, batch_size):
    """Evaluate model on test images and return accuracy."""
    try:
        predictions = model.predict(test_images, batch_size=batch_size, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(pred_labels == test_labels)
        return accuracy
    except Exception as e:
        print(f"ERROR during model evaluation: {e}")
        return None

def save_test_results_to_csv(model_name, accuracy, csv_file_path):
    """Append test results to CSV file."""
    file_exists = os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0
    
    try:
        with open(csv_file_path, 'a', newline='') as csvfile:
            fieldnames = ['model_file_name', 'accuracy_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'model_file_name': model_name,
                'accuracy_score': f"{accuracy:.6f}"
            })
        
        return True
    except Exception as e:
        print(f"ERROR saving results to CSV: {e}")
        return False


# ==================================================================================
# MODEL TRAINING
# ==================================================================================

def load_and_prepare_model(config):
    """Load model and set trainable layers."""
    model = tf.keras.models.load_model(config['MODEL_PATH'])
    
    trainable_layers = config.get('TRAINABLE_LAYERS', [])
    for idx, layer in enumerate(model.layers):
        if idx in trainable_layers:
            layer.trainable = True
            print(f"Training layer {idx}: {layer.name}")
        else:
            layer.trainable = False
    
    # Calculate and print trainable parameters
    trainable_params = 0
    total_params = 0
    
    for layer in model.layers:
        layer_params = layer.count_params()
        total_params += layer_params
        if layer.trainable:
            trainable_params += layer_params
    
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    return model

def create_data_generators(config):
    """Create training and validation data generators."""
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=False,
        validation_split=0.1,
        rescale=1.0 / 255)

    train_generator = datagen.flow_from_directory(
        config['BASE_DIR'],
        target_size=(config['IMAGE_SIZE'], config['IMAGE_SIZE']),
        batch_size=config['BATCH_SIZE'],
        subset='training',
        color_mode='rgb',
        class_mode='categorical')
    
    val_generator = datagen.flow_from_directory(
        config['BASE_DIR'],
        target_size=(config['IMAGE_SIZE'], config['IMAGE_SIZE']),
        batch_size=config['BATCH_SIZE'],
        subset='validation',
        color_mode='rgb',
        class_mode='categorical')
    
    return train_generator, val_generator

def compile_model(model, config):
    """Compile model with optimizer and loss."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['LEARNING_RATE']),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

def setup_callbacks(config):
    """Setup training callbacks (checkpoint and early stopping)."""
    os.makedirs('retrained_models', exist_ok=True)
    
    checkpoint_path = 'retrained_models/best_weights.h5'
    
    es_cfg = config.get('EARLY_STOPPING', {})
    es_monitor = es_cfg.get('MONITOR', 'val_accuracy')
    es_mode = es_cfg.get('MODE', 'auto')
    
    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=es_monitor,
        save_best_only=True,
        save_weights_only=True,
        mode='max' if es_mode == 'auto' and 'accuracy' in es_monitor else es_mode,
        verbose=0
    )
    
    callbacks = [checkpoint_cb]
    
    if es_cfg.get('ENABLED', False):
        earlystop_cb = EarlyStopping(
            monitor=es_monitor,
            patience=es_cfg.get('PATIENCE', 3),
            min_delta=float(es_cfg.get('MIN_DELTA', 0.0)),
            mode=es_mode,
            restore_best_weights=es_cfg.get('RESTORE_BEST_WEIGHTS', True),
            verbose=1
        )
        callbacks.append(earlystop_cb)
        print(f"EarlyStopping enabled: monitor={es_monitor}, patience={es_cfg.get('PATIENCE')}")
    else:
        print("EarlyStopping disabled")
    
    return callbacks, checkpoint_path

def train_model(model, train_gen, val_gen, config, callbacks):
    """Train the model."""
    history = model.fit(
        train_gen,
        epochs=config['EPOCHS'],
        validation_data=val_gen,
        verbose=1,
        callbacks=callbacks
    )
    
    return history

def save_model(model, history, config):
    """Save trained model with metrics in filename."""
    checkpoint_path = 'retrained_models/best_weights.h5'
    
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
    
    es_monitor = config.get('EARLY_STOPPING', {}).get('MONITOR', 'val_accuracy')
    
    hist = history.history
    if es_monitor in hist:
        best_mon_val = max(hist[es_monitor])
    elif 'val_accuracy' in hist:
        best_mon_val = max(hist['val_accuracy'])
    else:
        numeric_vals = []
        for v in hist.values():
            try:
                numeric_vals.extend([float(x) for x in v])
            except Exception:
                pass
        best_mon_val = max(numeric_vals) if numeric_vals else 0.0
    
    best_mon_str = f"{best_mon_val:.5f}".replace(".", "")
    today = datetime.datetime.now().strftime('%Y%m%d')
    
    save_path = f'retrained_models/vww_finetuned_{today}_{best_mon_str}.h5'
    model.save(save_path)
    print(f"Fine-tuned model saved to {save_path}")
    
    return save_path

def test_trained_model(model, config, model_path):
    """Test the trained model on test set and save results."""
    try:
        csv_file_path = os.path.join(config['TEST_OUTPUT_DIR'], 'model_accuracy_scores.csv')
        os.makedirs(config['TEST_OUTPUT_DIR'], exist_ok=True)
        
        test_images, test_labels, image_files = prepare_test_dataset(
            config['TEST_IMAGES_DIR'], 
            config['IMAGE_SIZE']
        )
        
        test_dirs = set(os.path.dirname(f) for f in image_files)
        test_dirs_str = ", ".join(sorted([os.path.basename(d) for d in test_dirs]))
        
        print(f"\nTesting model: {os.path.basename(model_path)}")
        print(f"Test images: {test_dirs_str} ({len(test_images)} images)")
        
        # Evaluate model
        test_accuracy = evaluate_model(model, test_images, test_labels, config['BATCH_SIZE'])
        
        if test_accuracy is not None:
            model_filename = os.path.basename(model_path)
            
            if save_test_results_to_csv(model_filename, test_accuracy, csv_file_path):
                print(f"Test score added to {csv_file_path}")
            else:
                print(f"ERROR: Could not add test score to CSV")
        else:
            print(f"ERROR: Could not evaluate model")
    
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except ValueError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"ERROR: {e}")

def save_training_summary(layer_idx, history, test_acc, model_path):
    csv_path = "model_scores.csv"
    file_exists = os.path.exists(csv_path)

    epochs_run = len(history.history["loss"])
    best_val_acc = max(history.history.get("val_accuracy", [0]))

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["layer_idx", "epochs_run", "best_val_accuracy", "test_accuracy", "model_path"])

        writer.writerow([
            layer_idx,
            epochs_run,
            f"{best_val_acc:.6f}",
            f"{test_acc:.6f}",
            os.path.basename(model_path)
        ])



# ==================================================================================
# MAIN
# ==================================================================================

def main(argv):
    try:
        config = load_config()
        validate_config(config)
        validate_paths(config)

        layer_list = config.get("TRAINABLE_LAYERS", [])
        if not layer_list:
            raise ValueError("TRAINABLE_LAYERS is empty in config.")

        print(f"Running {len(layer_list)} fine-tuning sessions.")

        for layer_idx in layer_list:
            print(f"\nStarting training for layer {layer_idx}")

            # Load a fresh model for each single-layer training run
            model = load_and_prepare_model({
                **config,
                "TRAINABLE_LAYERS": [layer_idx]
            })

            # Data
            train_gen, val_gen = create_data_generators(config)

            # Compile
            compile_model(model, config)

            # Callbacks
            callbacks, checkpoint_path = setup_callbacks(config)

            # Train
            history = train_model(model, train_gen, val_gen, config, callbacks)

            # Save model
            model_path = save_model(model, history, config)

            # Test
            test_images, test_labels, _ = prepare_test_dataset(
                config["TEST_IMAGES_DIR"],
                config["IMAGE_SIZE"]
            )
            test_acc = evaluate_model(model, test_images, test_labels, config["BATCH_SIZE"])

            # Record results
            save_training_summary(layer_idx, history, test_acc, model_path)

            print(f"Finished layer {layer_idx}. Test accuracy: {test_acc:.4f}")
            print(f"Model saved to: {model_path}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()



if __name__ == '__main__':
    app.run(main)