import os
import io
import json
import threading
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.efficientnet import preprocess_input
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
# Include application/octet-stream for mobile apps that send binary data without proper MIME type
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream"}
MAX_FILE_SIZE         = 10 * 1024 * 1024          # 10 MB
RETRAIN_THRESHOLD     = int(os.environ.get("RETRAIN_THRESHOLD", "10"))
MIN_CONFIDENCE         = float(os.environ.get("MIN_CONFIDENCE", "0.80"))
MIN_CONFIDENCE_HEALTHY = float(os.environ.get("MIN_CONFIDENCE_HEALTHY", "0.90"))
MIN_FEEDBACK_TO_TRAIN = 5                          # need at least this many images

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_DIR      = os.path.join(BASE_DIR, "feedback_data")
RETRAIN_MARKER    = os.path.join(BASE_DIR, "retrain_marker.json")
MODEL_PATH        = os.path.join(BASE_DIR, "best_plant_diagnosis.keras")
CLASS_NAMES_PATH  = os.path.join(BASE_DIR, "class_names.npy")

os.makedirs(FEEDBACK_DIR, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Plant Diagnosis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Model (mutable so hot-reload can swap it) ─────────────────────────────────
print("Loading disease model...")
MODEL = tf.keras.models.load_model(MODEL_PATH)
class_names: list[str] = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()
print(f"Disease model ready — {len(class_names)} classes.")

GATE_MODEL_PATH = os.path.join(BASE_DIR, "gate_model.keras")
GATE_THRESHOLD  = float(os.environ.get("GATE_THRESHOLD", "0.60"))
if os.path.exists(GATE_MODEL_PATH):
    print("Loading gate model...")
    print(f"[DEBUG] Gate model path: {GATE_MODEL_PATH}")
    print(f"[DEBUG] Gate model last modified: {datetime.fromtimestamp(os.path.getmtime(GATE_MODEL_PATH))}")
    GATE_MODEL = tf.keras.models.load_model(GATE_MODEL_PATH)
    print(f"Gate model ready (threshold: {GATE_THRESHOLD}).")
else:
    GATE_MODEL = None
    print("No gate model found — run train_gate.py to enable it.")

# ── Retraining state ──────────────────────────────────────────────────────────
_retrain_lock  = threading.Lock()
_is_retraining = False
_retrain_log: list[dict] = []


def _load_marker() -> int:
    """Return the feedback count at the time of the last retrain."""
    if os.path.exists(RETRAIN_MARKER):
        with open(RETRAIN_MARKER) as f:
            return json.load(f).get("count_at_last_retrain", 0)
    return 0


def _save_marker(count: int) -> None:
    with open(RETRAIN_MARKER, "w") as f:
        json.dump({"count_at_last_retrain": count, "timestamp": datetime.now().isoformat()}, f)


def _count_feedback() -> int:
    total = 0
    for cls_dir in os.listdir(FEEDBACK_DIR):
        path = os.path.join(FEEDBACK_DIR, cls_dir)
        if os.path.isdir(path):
            total += len(os.listdir(path))
    return total


def _load_feedback_dataset():
    """Load all saved feedback images into numpy arrays."""
    images, labels = [], []
    for cls_name in os.listdir(FEEDBACK_DIR):
        if cls_name not in class_names:
            continue
        cls_idx = class_names.index(cls_name)
        cls_path = os.path.join(FEEDBACK_DIR, cls_name)
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize((224, 224))
                images.append(np.array(img))
                one_hot = np.zeros(len(class_names), dtype=np.float32)
                one_hot[cls_idx] = 1.0
                labels.append(one_hot)
            except Exception:
                continue
    if not images:
        return None, None
    X = preprocess_input(np.array(images, dtype=np.float32))
    y = np.array(labels, dtype=np.float32)
    return X, y


def _log(event: str) -> None:
    _retrain_log.append({"timestamp": datetime.now().isoformat(), "event": event})


def _run_retraining() -> None:
    """Fine-tune the current model on accumulated feedback data (runs in a thread)."""
    global MODEL, _is_retraining

    _log("Fine-tuning started")
    print("[Retrain] Starting fine-tuning on feedback data...")

    try:
        X, y = _load_feedback_dataset()
        if X is None or len(X) < MIN_FEEDBACK_TO_TRAIN:
            _log("Skipped — not enough feedback images yet.")
            return

        new_model = tf.keras.models.load_model(MODEL_PATH)
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        ]

        new_model.fit(X, y, epochs=10, batch_size=16, callbacks=callbacks, verbose=0)
        new_model.save(MODEL_PATH)

        # Hot-reload: swap the global model atomically
        MODEL = new_model

        total = _count_feedback()
        _save_marker(total)

        _log(f"Fine-tuning complete on {len(X)} images. Model hot-reloaded.")
        print(f"[Retrain] Fine-tuning complete on {len(X)} images. Model hot-reloaded.")

    except Exception as exc:
        _log(f"Fine-tuning failed: {exc}")
        print(f"[Retrain] Failed: {exc}")
    finally:
        _is_retraining = False


# ── Helpers ───────────────────────────────────────────────────────────────────
def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def _preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.expand_dims(np.array(img), axis=0)
    return preprocess_input(arr.astype(np.float32))


def _validate_upload(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        print(f"[DEBUG] VALIDATION FAILED - Content-Type: {file.content_type}, Allowed: {ALLOWED_CONTENT_TYPES}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG, PNG, WEBP, or BMP image.",
        )


# ── Static data ───────────────────────────────────────────────────────────────
disease_advice = {
    "Apple___Apple_scab": "Use resistant varieties and apply fungicide sprays.",
    "Apple___Black_rot": "Prune infected branches and avoid overhead irrigation.",
    "Apple___Cedar_apple_rust": "Remove nearby junipers and apply fungicide.",
    "Apple___healthy": "No disease detected. Keep monitoring your plant.",
    "Tomato___Bacterial_spot": "Use copper sprays, avoid overhead watering.",
    "Tomato___Early_blight": "Remove infected leaves, apply fungicide, rotate crops.",
    "Tomato___Late_blight": "Destroy infected plants, use copper-based fungicides.",
    "Tomato___Leaf_Mold": "Improve ventilation, apply fungicide.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves, apply fungicide.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use miticides or insecticidal soap.",
    "Tomato___Target_Spot": "Apply fungicide, rotate crops.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies, remove infected plants.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants, disinfect tools.",
    "Tomato___healthy": "No disease detected. Keep monitoring your plant.",
    "fruit_apple_fresh": "Apple looks fresh and edible.",
    "fruit_apple_rotten": "Apple is rotten. Do not consume.",
    "fruit_tomato_fresh": "Tomato looks fresh and edible.",
    "fruit_tomato_rotten": "Tomato is rotten. Do not consume.",
}

status_mapping = {
    "fruit_apple_fresh": "Fresh Apple Fruit",
    "fruit_apple_rotten": "Rotten Apple Fruit",
    "fruit_tomato_fresh": "Fresh Tomato Fruit",
    "fruit_tomato_rotten": "Rotten Tomato Fruit",
    "Apple___Apple_scab": "Apple Leaf – Scab",
    "Apple___Black_rot": "Apple Leaf – Black Rot",
    "Apple___Cedar_apple_rust": "Apple Leaf – Cedar Rust",
    "Apple___healthy": "Healthy Apple Leaf",
    "Tomato___Bacterial_spot": "Tomato Leaf – Bacterial Spot",
    "Tomato___Early_blight": "Tomato Leaf – Early Blight",
    "Tomato___Late_blight": "Tomato Leaf – Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf – Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato Leaf – Septoria Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato Leaf – Spider Mites",
    "Tomato___Target_Spot": "Tomato Leaf – Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Leaf – Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato Leaf – Mosaic Virus",
    "Tomato___healthy": "Healthy Tomato Leaf",
}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    print(f"[DEBUG] ========== HEALTH CHECK REQUEST ==========")
    return {"status": "ok", "classes": len(class_names)}


@app.get("/classes")
def get_classes():
    print(f"[DEBUG] ========== CLASSES REQUEST ==========")
    return {"classes": class_names}


@app.get("/retrain/status")
def retrain_status():
    total      = _count_feedback()
    since_last = total - _load_marker()
    return {
        "status":             "Training in Progress" if _is_retraining else "Model is Idle",
        "progress":           since_last,
        "threshold":          RETRAIN_THRESHOLD,
        "total_samples":      total,
        "since_last_retrain": since_last,
        "log":                _retrain_log[-10:],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(f"[DEBUG] ========== PREDICT REQUEST RECEIVED ==========")
    print(f"[DEBUG] Filename: {file.filename}, Content-Type: {file.content_type}")
    
    _validate_upload(file)
    image_bytes = await file.read()
    print(f"[DEBUG] Received upload: filename={file.filename}, content_type={file.content_type}, size={len(image_bytes)} bytes")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")

    try:
        processed = _preprocess(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read the image. Please upload a valid image file.")

    # ── Gate check ────────────────────────────────────────────────────────────
    # Runs BEFORE the disease model. If the gate says "not a plant", reject
    # immediately without running the more expensive disease classifier.
    if GATE_MODEL is not None:
        gate_score = float(GATE_MODEL.predict(processed, verbose=0)[0][0])
        print(f"[DEBUG] gate_score = {gate_score:.4f}  (threshold = {GATE_THRESHOLD})")
        if gate_score < GATE_THRESHOLD:
            print(f"[DEBUG] REJECTED at gate stage (score {gate_score:.4f} < {GATE_THRESHOLD})")
            return {
                "success":    True,
                "message":    "This image does not appear to be an apple or tomato leaf or fruit. Please take a clear, close-up photo of the plant.",
                "status":     "Not Recognised",
                "confidence": f"{gate_score * 100:.2f}%",
                "danger":     "N/A",
                "advice":     "Point the camera directly at a single leaf or piece of fruit. Make sure it fills the frame and is in focus.",
                "warning":    "This model only recognises apple and tomato leaves and fruit.",
                "class_name": "unknown",
            }

    prediction  = MODEL.predict(processed, verbose=0)
    class_index = int(prediction.argmax())
    confidence  = float(prediction[0][class_index])
    class_name  = class_names[class_index]
    print(f"[DEBUG] disease prediction = {class_name}, confidence = {confidence:.4f}")

    # Secondary confidence check — catches edge cases the gate passed
    # but the disease model itself isn't sure about.
    is_healthy_class = "healthy" in class_name or "fresh" in class_name
    threshold = MIN_CONFIDENCE_HEALTHY if is_healthy_class else MIN_CONFIDENCE
    print(f"[DEBUG] threshold applied = {threshold} (is_healthy_class={is_healthy_class})")

    if confidence < threshold:
        print(f"[DEBUG] REJECTED at disease-confidence stage (confidence {confidence:.4f} < {threshold})")
        return {
            "success":    True,
            "message":    "This image does not look like an apple or tomato leaf or fruit. Please take a clear, close-up photo of the plant.",
            "status":     "Not Recognised",
            "confidence": f"{confidence * 100:.2f}%",
            "danger":     "N/A",
            "advice":     "Point the camera at a single apple or tomato leaf, or at the fruit itself. Make sure the image is in focus and well-lit.",
            "warning":    "Image not recognised. This model is trained only for apple and tomato diseases.",
            "class_name": "unknown",
        }

    advice  = disease_advice.get(class_name, "No advice available.")
    warning = None
    if confidence < 0.85:
        warning = f"Moderate confidence ({confidence * 100:.2f}%). For best results, use a clear, close-up photo."

    if class_name.startswith("fruit_"):
        _, fruit, condition = class_name.split("_", 2)
        fruit_cap = fruit.capitalize()
        if condition == "rotten":
            message = f"This is {_article(fruit_cap)} {fruit_cap} fruit. It appears rotten, possibly due to fungal or bacterial infection."
            danger  = "Eating rotten fruit can cause food poisoning, stomach upset, or infections."
            advice  = "Do not consume. If already eaten, monitor for nausea, vomiting, or diarrhea. Seek medical attention if symptoms appear."
        else:
            message = f"This is {_article(fruit_cap)} {fruit_cap} fruit. It looks fresh and safe to eat."
            danger  = "No immediate danger detected."
    elif "Apple" in class_name or "Tomato" in class_name:
        plant, disease = class_name.split("___")
        message = f"This is {_article(plant)} {plant} leaf. Disease: {disease.replace('_', ' ').capitalize()}."
        danger  = "Leaf diseases reduce crop yield but are not directly harmful if leaves are not eaten."
    else:
        message = "Unknown input. Please upload a valid apple or tomato fruit/leaf image."
        danger  = "Unknown risk."
        advice  = "Please provide a clearer image of an apple or tomato leaf or fruit."

    return {
        "success":    True,
        "message":    message,
        "status":     status_mapping.get(class_name, class_name),
        "confidence": f"{confidence * 100:.2f}%",
        "danger":     danger,
        "advice":     advice,
        "warning":    warning,
        "class_name": class_name,   # raw name, useful for the feedback form
    }


@app.post("/feedback")
async def feedback(file: UploadFile = File(...), label: str = Form(...)):
    """
    Accept a user-confirmed label for an image.
    Saves the image and triggers retraining automatically once the threshold is reached.
    """
    global _is_retraining
    
    print(f"[DEBUG] ========== FEEDBACK REQUEST RECEIVED ==========")
    print(f"[DEBUG] Filename: {file.filename}, Label: {label}, Content-Type: {file.content_type}")

    if label not in class_names:
        raise HTTPException(status_code=400, detail=f"Unknown label '{label}'. Call GET /classes for valid options.")

    _validate_upload(file)
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read the image.")

    # Save to feedback_data/<label>/<timestamp>.jpg
    label_dir = os.path.join(FEEDBACK_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img.save(os.path.join(label_dir, f"{timestamp}.jpg"), "JPEG")

    total      = _count_feedback()
    since_last = total - _load_marker()
    triggered  = False

    if since_last >= RETRAIN_THRESHOLD and not _is_retraining:
        with _retrain_lock:
            if not _is_retraining:           # double-check inside lock
                _is_retraining = True
                threading.Thread(target=_run_retraining, daemon=True).start()
                triggered = True

    return {
        "success":          True,
        "message":          f"Feedback saved. {since_last}/{RETRAIN_THRESHOLD} images collected since last retrain.",
        "total_feedback":   total,
        "retrain_triggered": triggered,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)