import os
import datetime
import numpy as np
from absl import app
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def load_config(config_path='retrain_config.yaml'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found. Please provide a valid config file.")
    with open(config_path, 'r') as f:
        user_config = yaml.safe_load(f)
        if not user_config:
            raise ValueError(f"Config file '{config_path}' is empty or invalid.")
    return user_config

def main(argv):
    config = load_config()
    MODEL_PATH = config["MODEL_PATH"]
    IMAGE_SIZE = config["IMAGE_SIZE"]
    BATCH_SIZE = config["BATCH_SIZE"]
    EPOCHS = config["EPOCHS"]
    BASE_DIR = config["BASE_DIR"]
    LEARNING_RATE = config["LEARNING_RATE"]

    trainable_layers = config.get("TRAINABLE_LAYERS", [61, 67])

    es_cfg = config.get("EARLY_STOPPING", {})
    es_enabled = bool(es_cfg.get("ENABLED", False))
    es_patience = int(es_cfg.get("PATIENCE", 3))
    es_monitor = es_cfg.get("MONITOR", "val_accuracy")
    es_min_delta = float(es_cfg.get("MIN_DELTA", 0.0))
    es_mode = es_cfg.get("MODE", "auto")
    es_restore_best = bool(es_cfg.get("RESTORE_BEST_WEIGHTS", True))

    model = tf.keras.models.load_model(MODEL_PATH)

    for idx, layer in enumerate(model.layers):
        if idx in trainable_layers:
            layer.trainable = True
            print(f"Training layer {idx}: {layer.name}")
        else:
            layer.trainable = False
    
    #model.summary()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        validation_split=0.1,
        rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='training',
        color_mode='rgb',
        class_mode='categorical')
    
    val_generator = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        color_mode='rgb',
        class_mode='categorical')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Ensure output directory exists
    os.makedirs('retrained_models', exist_ok=True)

    # Save best weights during training (monitored metric comes from config)
    checkpoint_path = 'retrained_models/best_weights.h5'
    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=es_monitor,
        save_best_only=True,
        save_weights_only=True,
        mode='max' if es_mode == 'auto' and 'accuracy' in es_monitor else es_mode,
        verbose=0
    )

    # Build callbacks list and include EarlyStopping if enabled
    callbacks = [checkpoint_cb]

    if es_enabled:
        earlystop_cb = EarlyStopping(
            monitor=es_monitor,
            patience=es_patience,
            min_delta=es_min_delta,
            mode=es_mode,
            restore_best_weights=es_restore_best,
            verbose=1
        )
        callbacks.append(earlystop_cb)
        print(f"EarlyStopping enabled: monitor={es_monitor}, patience={es_patience}, min_delta={es_min_delta}, mode={es_mode}, restore_best_weights={es_restore_best}")
    else:
        print("EarlyStopping disabled")

    # Train with callbacks
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=1,
        callbacks=callbacks
    )

    # Restore best weights (ModelCheckpoint saved weights)
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    # Determine best monitored metric value for naming the file
    monitor_key = es_monitor
    hist = history.history
    if monitor_key in hist:
        best_mon_val = max(hist[monitor_key])
    elif 'val_accuracy' in hist:
        best_mon_val = max(hist['val_accuracy'])
    else:
        # fallback if requested metric not present
        numeric_vals = []
        for v in hist.values():
            try:
                numeric_vals.extend([float(x) for x in v])
            except Exception:
                pass
        best_mon_val = max(numeric_vals) if numeric_vals else 0.0

    best_mon_str = f"{best_mon_val:.5f}".replace(".", "")  # strip dot for filename
    today = datetime.datetime.now().strftime('%Y%m%d')

    # Create unique filename
    save_path = f'retrained_models/vww_finetuned_{today}_{best_mon_str}.h5'
    model.save(save_path)
    print(f"Fine-tuned model saved to {save_path}")

if __name__ == '__main__':
    app.run(main)