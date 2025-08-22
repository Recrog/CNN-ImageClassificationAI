
# Robust Traffic Sign Training (Fixed & Consistent)
# robust_training_fixed.py
# - Consistent 32x32 input to match Streamlit app
# - Fixed imports (uses tf.keras properly)
# - Albumentations without conflicting Normalize for the CNN path
# - Proper per-channel standardization computed from train and reused everywhere
# - Safe augmentation layers (no RandomBrightness/Contrast that may be missing)
# - CLI args instead of interactive input()
# - Saves best model as .keras + exports channel stats for the app

import os
import cv2
import json
import pickle
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import albumentations as A
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Constants ----------
NUM_CLASSES = 43
IMG_SIZE = 32  # Keep 32x32 to match web app
CHANNELS = 3

# Fallback paths (handles the misspelling in folder name)
POSSIBLE_zDIRS = [
    "Trafic Signs Preprocssed data/",
    "./Trafic Signs Preprocssed data/",
    "Traffic Signs Preprocessed data/",
    "./Traffic Signs Preprocessed data/",
]

# ---------- Data Loading & Preprocess ----------
def _find_data_dir():
    for d in POSSIBLE_zDIRS:
        if os.path.exists(os.path.join(d, "train.pickle")):
            return d
    raise FileNotFoundError("Veri dosyaları bulunamadı! (train/valid/test pickle). Klasör adını kontrol edin.")

def _resize_stack(X, size=IMG_SIZE):
    # Handles grayscale or RGBA by converting to RGB first
    X_out = []
    for img in X:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img32 = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        X_out.append(img32)
    return np.array(X_out, dtype=np.uint8)

def load_preprocessed_data():
    data_dir = _find_data_dir()
    logging.info(f"📦 Veri klasörü: {data_dir}")
    with open(os.path.join(data_dir, "train.pickle"), "rb") as f:
        tr = pickle.load(f)
    with open(os.path.join(data_dir, "valid.pickle"), "rb") as f:
        va = pickle.load(f)
    with open(os.path.join(data_dir, "test.pickle"), "rb") as f:
        te = pickle.load(f)

    X_train = _resize_stack(tr["features"], IMG_SIZE)
    X_valid = _resize_stack(va["features"], IMG_SIZE)
    X_test  = _resize_stack(te["features"], IMG_SIZE)
    y_train, y_valid, y_test = tr["labels"], va["labels"], te["labels"]

    logging.info(f"✅ Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def compute_channel_stats(X_uint8):
    X = X_uint8.astype("float32") / 255.0
    means = [float(X[:,:,:,c].mean()) for c in range(CHANNELS)]
    stds  = [float(X[:,:,:,c].std() + 1e-8) for c in range(CHANNELS)]
    return means, stds

def standardize_batch(X_float01, means, stds):
    Xs = X_float01.copy()
    for c in range(CHANNELS):
        Xs[:,:,:,c] = (Xs[:,:,:,c] - means[c]) / stds[c]
    return Xs

# ---------- Augmentations ----------
def create_albu_transform():
    # Real-world-ish but not destructive; output stays in uint8 [0,255]
    return A.Compose([
        A.Rotate(limit=15, p=0.6),
        A.RandomScale(scale_limit=0.12, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.15),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=12, p=0.3),
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_AREA)
    ])

def aug_flow_uint8(X_uint8, y, batch_size, means, stds, transform):
    # Takes uint8 images, augments, then normalizes to 0-1 and standardizes with (means,stds)
    n = len(X_uint8)
    idx = np.arange(n)
    while True:
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            sel = idx[start:start+batch_size]
            Xb = X_uint8[sel]
            yb = y[sel]
            X_aug = np.empty((len(sel), IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.float32)
            for i, img in enumerate(Xb):
                aug = transform(image=img)["image"]
                X_aug[i] = aug.astype("float32") / 255.0
            X_aug = standardize_batch(X_aug, means, stds)
            yield X_aug, yb

# ---------- Model ----------
def create_robust_cnn(input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS), num_classes=NUM_CLASSES):
    aug_layer = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        # Use a small GaussianNoise from Keras (safe & available)
        layers.GaussianNoise(0.02),
    ], name="keras_augmentation")

    model = keras.Sequential([
        aug_layer,

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.35),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="robust_cnn32")
    return model

def compile_model(model, lr=1e-3):
    # AdamW is available in newer TF; fallback to Adam if not
    try:
        opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    except Exception:
        opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def setup_callbacks(model_name="robust_traffic_signs"):
    return [
        EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=5e-6, verbose=1),
        ModelCheckpoint(f"{model_name}_best.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

# ---------- Evaluation ----------
def create_challenging_versions(img_uint8):
    # Returns a list of uint8 variants at 32x32
    img = cv2.resize(img_uint8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    out = []

    out.append(cv2.GaussianBlur(img, (3,3), 0))
    out.append(np.clip(img.astype(np.float32) * 0.6, 0, 255).astype(np.uint8))  # dark
    out.append(np.clip(img.astype(np.float32) * 1.4, 0, 255).astype(np.uint8))  # bright

    noise = np.random.normal(0, 12, img.shape).astype(np.int16)
    out.append(np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8))

    M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), 7, 1.0)
    out.append(cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT_101))
    return out

def build_challenging_set(X_uint8, y, max_src=1000):
    Xc, yc = [], []
    limit = min(max_src, len(X_uint8))
    for i in range(limit):
        for v in create_challenging_versions(X_uint8[i]):
            Xc.append(v)
            yc.append(y[i])
    return np.array(Xc, dtype=np.uint8), np.array(yc)

def evaluate(model, X_float_std, y, tag="SET"):
    preds = model.predict(X_float_std, verbose=0)
    y_hat = np.argmax(preds, axis=1)
    print(f"\n📋 Classification report ({tag}):")
    print(classification_report(y, y_hat, digits=4))

# ---------- Main ----------
def main(model_name="robust_cnn32", epochs=50, batch_size=32, seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 1) Load & resize
    Xtr_u8, ytr, Xva_u8, yva, Xte_u8, yte = load_preprocessed_data()

    # 2) Compute channel stats from TRAIN (0..1)
    means, stds = compute_channel_stats(Xtr_u8)
    logging.info(f"📐 Channel means: {means}")
    logging.info(f"📐 Channel stds : {stds}")

    # Save stats for the web app
    with open("channel_stats.json", "w", encoding="utf-8") as f:
        json.dump({"means": means, "stds": stds}, f, ensure_ascii=False, indent=2)

    # 3) Prepare validation/test (float in [0,1] then standardize)
    Xva = standardize_batch(Xva_u8.astype("float32")/255.0, means, stds)
    Xte = standardize_batch(Xte_u8.astype("float32")/255.0, means, stds)

    # 4) Model
    model = create_robust_cnn()
    model = compile_model(model, lr=1e-3)
    model.summary()

    # 5) Training with Albumentations generator (from uint8)
    albu = create_albu_transform()
    steps = max(1, len(Xtr_u8)//batch_size)
    callbacks = setup_callbacks(model_name)

    history = model.fit(
        aug_flow_uint8(Xtr_u8, ytr, batch_size, means, stds, albu),
        steps_per_epoch=steps,
        epochs=epochs,
        validation_data=(Xva, yva),
        callbacks=callbacks,
        verbose=1
    )

    # 6) Normal test
    test_loss, test_acc = model.evaluate(Xte, yte, verbose=0)
    print(f"\n🧪 Normal Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # 7) Challenging test
    Xc_u8, yc = build_challenging_set(Xte_u8, yte, max_src=600)  # use up to 600 images for speed
    Xc = standardize_batch(Xc_u8.astype('float32')/255.0, means, stds)
    ch_loss, ch_acc = model.evaluate(Xc, yc, verbose=0)
    print(f"💪 Challenging Test Accuracy: {ch_acc:.4f} ({ch_acc*100:.2f}%)")
    print(f"🧭 Robustness Ratio: {(ch_acc/(test_acc+1e-8))*100:.2f}%")

    # 8) Detailed reports
    evaluate(model, Xte, yte, tag="NORMAL TEST")
    evaluate(model, Xc, yc, tag="CHALLENGING TEST")

    # 9) Save final model (timestamped) + best already saved via checkpoint
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_name = f"{model_name}_{stamp}.keras"
    model.save(final_name)
    print(f"\n💾 Saved: {final_name}")
    print("💾 Best checkpoint: {}_best.keras".format(model_name))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="robust_cnn32")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    main(model_name=args.model_name, epochs=args.epochs, batch_size=args.batch_size)
