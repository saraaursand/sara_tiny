import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import csv
from pathlib import Path
from datetime import datetime

# Configuration (matches retrain_config.yaml)
MODELS_DIR = r"C:\Users\Sara\ProjectThesis\sara_tiny\TinyMobileNet\retrained_models"
TEST_IMAGES_DIR = r"C:\Users\Sara\ProjectThesis\sara_tiny\test_images\zero_not_zero"
OUTPUT_DIR = r"C:\Users\Sara\ProjectThesis\sara_tiny\model_testig\tiny"
IMAGE_SIZE = 96  # From retrain_config.yaml
BATCH_SIZE = 32  # From retrain_config.yaml

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_image_files(directory):
    """Get all image files from the test directory recursively."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    
    return sorted(image_files)

def get_labels_from_structure(test_dir):
    """
    Extract labels from directory structure.
    Assumes structure: test_dir/class_name/images
    For 'zero_not_zero', expects: zero/ and not_zero/ subdirectories
    """
    labels_dict = {}
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                # Get the class label from the parent directory name
                class_label = os.path.basename(root)
                labels_dict[file_path] = class_label
    
    return labels_dict

def load_and_preprocess_image(image_path, image_size=IMAGE_SIZE):
    """
    Load and preprocess a single image to match training preprocessing.
    Matches the ImageDataGenerator(rescale=1./255) from training.
    """
    img = load_img(image_path, target_size=(image_size, image_size), color_mode='rgb')
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to 0-1 range (matching training)
    return img_array

def prepare_test_dataset(test_images_dir, image_size=IMAGE_SIZE):
    """
    Prepare test dataset from directory structure.
    Returns: images array, file paths, and true labels
    """
    image_files = get_image_files(test_images_dir)
    
    if not image_files:
        raise ValueError(f"No images found in {test_images_dir}")
    
    print(f"Found {len(image_files)} test images")
    
    # Get labels from directory structure
    labels_dict = get_labels_from_structure(test_images_dir)
    
    # Prepare unique class labels (sorted for consistency)
    unique_classes = sorted(set(labels_dict.values()))
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    images = []
    true_labels = []
    processed_files = []
    
    for img_path in image_files:
        try:
            img_array = load_and_preprocess_image(img_path, image_size)
            images.append(img_array)
            true_labels.append(class_to_idx[labels_dict[img_path]])
            processed_files.append(img_path)
        except Exception as e:
            print(f"Warning: Skipping {os.path.basename(img_path)}: {e}")
            continue
    
    images = np.array(images)
    true_labels = np.array(true_labels)
    
    print(f"Successfully processed {len(images)} images")
    print(f"Classes found: {unique_classes}")
    print(f"Class mapping: {class_to_idx}")
    
    return images, true_labels, processed_files, unique_classes

def evaluate_model(model, images, true_labels):
    """
    Evaluate model on test images and compute accuracy.
    Handles categorical classification output.
    """
    try:
        # Make predictions
        predictions = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
        
        # Get predicted class indices (argmax for categorical output)
        pred_labels = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(pred_labels == true_labels)
        
        return accuracy
    except Exception as e:
        print(f"  ERROR during evaluation: {e}")
        return None

def get_model_files(models_dir):
    """Get all .h5 model files from the models directory."""
    model_files = glob.glob(os.path.join(models_dir, '*.h5'))
    return sorted(model_files)

def main():
    print("\n" + "=" * 90)
    print(" " * 20 + "TinyMobileNet Model Testing Script")
    print("=" * 90)
    print(f"Models directory:      {MODELS_DIR}")
    print(f"Test images directory: {TEST_IMAGES_DIR}")
    print(f"Output directory:      {OUTPUT_DIR}")
    print(f"Image size:            {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size:            {BATCH_SIZE}")
    print("=" * 90 + "\n")
    
    # Get model files
    model_files = get_model_files(MODELS_DIR)
    
    if not model_files:
        print(f"ERROR: No .h5 models found in {MODELS_DIR}")
        print("Please ensure your models are saved as .h5 files")
        return
    
    print(f"Found {len(model_files)} model(s) to test:")
    for idx, model_file in enumerate(model_files, 1):
        print(f"  {idx}. {os.path.basename(model_file)}")
    print()
    
    # Prepare test dataset
    print("-" * 90)
    print("Preparing test dataset...")
    print("-" * 90)
    
    try:
        images, true_labels, image_paths, classes = prepare_test_dataset(TEST_IMAGES_DIR, IMAGE_SIZE)
        print(f"Dataset shape: {images.shape}")
        print(f"Total images: {len(true_labels)}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to prepare test dataset: {e}")
        return
    
    # Evaluate each model
    results = []
    
    print("-" * 90)
    print("Evaluating models...")
    print("-" * 90 + "\n")
    
    for i, model_path in enumerate(model_files, 1):
        model_name = os.path.basename(model_path)
        print(f"[{i}/{len(model_files)}] Testing: {model_name}")
        
        try:
            # Load model
            print(f"  Loading model...", end=" ")
            model = tf.keras.models.load_model(model_path)
            print("✓")
            
            # Evaluate
            print(f"  Evaluating on {len(images)} test images...", end=" ")
            accuracy = evaluate_model(model, images, true_labels)
            
            if accuracy is not None:
                percentage = accuracy * 100
                print(f"✓")
                print(f"  → Accuracy: {accuracy:.6f} ({percentage:.2f}%)\n")
                results.append({
                    'model_file_name': model_name,
                    'accuracy_score': f"{accuracy:.6f}"
                })
            else:
                print(f"✗ ERROR")
                results.append({
                    'model_file_name': model_name,
                    'accuracy_score': 'ERROR'
                })
            
            # Free memory
            del model
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"✗ ERROR: {e}\n")
            results.append({
                'model_file_name': model_name,
                'accuracy_score': 'ERROR'
            })
    
    # Save results to CSV
    csv_file_path = os.path.join(OUTPUT_DIR, 'model_accuracy_scores.csv')
    
    print("-" * 90)
    print("Saving results to CSV...")
    print("-" * 90)
    
    try:
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['model_file_name', 'accuracy_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"✓ Results saved to: {csv_file_path}\n")
        
        # Print summary
        print("=" * 90)
        print(" " * 30 + "TEST RESULTS SUMMARY")
        print("=" * 90)
        print(f"{'Model Name':<50} {'Accuracy Score':<20}")
        print("-" * 90)
        for result in results:
            print(f"{result['model_file_name']:<50} {result['accuracy_score']:<20}")
        print("=" * 90 + "\n")
        
    except Exception as e:
        print(f"ERROR: Failed to save CSV file: {e}")
    
    print("✓ Test script completed successfully!")

if __name__ == '__main__':
    main()