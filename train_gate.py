
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE          = 224
SAMPLES_PER_CLASS = 1500       # 1500 plants + 1500 non-plants = 3000 total
BATCH_SIZE        = 32
EPOCHS            = 15

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
GATE_MODEL_PATH = os.path.join(BASE_DIR, "gate_model.keras")


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_plant_images(n: int) -> np.ndarray:
    print(f"  Scanning {DATASET_DIR} ...")
    paths = []
    for cls_name in os.listdir(DATASET_DIR):
        cls_path = os.path.join(DATASET_DIR, cls_name)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            paths.append(os.path.join(cls_path, fname))

    random.shuffle(paths)
    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.float32))
        except Exception:
            continue
        if len(images) == n:
            break

    print(f"  Loaded {len(images)} plant images.")
    return np.array(images)


def load_negative_images(n: int) -> np.ndarray:
    """
    Real, full-resolution photos of everyday non-plant objects.
    Uses Imagenette (a small ImageNet subset: fish, chainsaws, churches,
    gas pumps, golf balls, parachutes, garbage trucks, etc.) so the gate
    learns actual content differences instead of a resolution/blur
    shortcut that CIFAR-10 introduces.
    """
    import tensorflow_datasets as tfds

    print("  Loading Imagenette (downloads ~1.5 GB on first run)...")
    ds = tfds.load("imagenette/320px-v2", split="train", as_supervised=True)

    images = []
    for img_tensor, _ in ds.take(n * 2):   # take extra as a buffer
        img = Image.fromarray(img_tensor.numpy()).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
        images.append(np.array(img, dtype=np.float32))
        if len(images) == n:
            break

    print(f"  Loaded {len(images)} non-plant images (Imagenette).")
    return np.array(images)


# ── Load data ───────────────────────────────────────────────────────────────
print("\n── Positive samples (plants) ────────────────────────────────")
positives = load_plant_images(SAMPLES_PER_CLASS)

print("\n── Negative samples (non-plants) ────────────────────────────")
negatives = load_negative_images(SAMPLES_PER_CLASS)

# ── Build dataset ────────────────────────────────────────────────────────────
X = preprocess_input(np.concatenate([positives, negatives]))
y = np.array(
    [1.0] * len(positives) + [0.0] * len(negatives),
    dtype=np.float32,
)

perm   = np.random.permutation(len(X))
X, y   = X[perm], y[perm]

split          = int(0.85 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"\nDataset ready — train: {len(X_train)}  |  val: {len(X_val)}")

# ── Model ─────────────────────────────────────────────────────────────────
print("\n── Building gate model ─────────────────────────────────────")
base = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False          # freeze backbone — only train the head

gate = tf.keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),   # 1 = plant, 0 = not plant
], name="plant_gate")

gate.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# ── Train ────────────────────────────────────────────────────────────────────
print("\n── Training ────────────────────────────────────────────────")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        GATE_MODEL_PATH, save_best_only=True, monitor="val_accuracy"
    ),
]

gate.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

_, val_acc = gate.evaluate(X_val, y_val, verbose=0)
print(f"\nGate model saved → {GATE_MODEL_PATH}")
print(f"Validation accuracy: {val_acc * 100:.2f}%")
print("\nRestart prediction.py — it will load the gate automatically.")