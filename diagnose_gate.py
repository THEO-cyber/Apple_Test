import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

IMG_SIZE = 224
GATE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_model.keras")


def main(folder: str):
    print(f"Loading gate model from {GATE_MODEL_PATH} ...")
    gate = tf.keras.models.load_model(GATE_MODEL_PATH)

    scores = []
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        except Exception:
            continue
        arr = preprocess_input(np.expand_dims(np.array(img, dtype=np.float32), axis=0))
        score = float(gate.predict(arr, verbose=0)[0][0])
        scores.append(score)
        print(f"{fname:40s}  gate_score = {score:.4f}")

    if scores:
        scores = np.array(scores)
        print("\n── Summary ──────────────────────────────")
        print(f"  n            = {len(scores)}")
        print(f"  mean score   = {scores.mean():.4f}")
        print(f"  min score    = {scores.min():.4f}")
        print(f"  max score    = {scores.max():.4f}")
        print(f"  % below 0.60 = {(scores < 0.60).mean() * 100:.1f}%  (current default threshold)")
        print(f"  % below 0.40 = {(scores < 0.40).mean() * 100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_gate.py /path/to/folder")
        sys.exit(1)
    main(sys.argv[1])