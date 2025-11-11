import os
import numpy as np
import cv2
import tensorflow as tf

# --------- User Configuration ---------
MODEL_PATH = "retrained_models/vww_finetuned_20251014_099317.h5" 
IMAGES_DIR = "test_images/zero_not_zero"  # Folder with images to verify
IMAGE_SIZE = 96
TRAIN_DATA_DIR = "retrain_images/mnist"  # Folder used for training (to get class names)
# --------------------------------------

# Get class names from training directory
class_names = sorted([
    d for d in os.listdir(TRAIN_DATA_DIR)
    if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))
])

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

for img_file in os.listdir(IMAGES_DIR):
    img_path = os.path.join(IMAGES_DIR, img_file)
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    input_img = preprocess_image(img_path)
    preds = model.predict(input_img)
    pred_class_idx = np.argmax(preds)
    pred_label = class_names[pred_class_idx]

    # Show image with predicted label
    img_disp = cv2.imread(img_path)
    cv2.putText(img_disp, pred_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Prediction', img_disp)
    print(f"{img_file}: {pred_label}")
    cv2.waitKey(0)

cv2.destroyAllWindows()