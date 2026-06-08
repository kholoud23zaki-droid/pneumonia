import os
import threading
import numpy as np
from PIL import Image
from contextlib import asynccontextmanager

import gdown
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "pneumonia_model.keras"
GDRIVE_FILE_ID = "1p0dewjOLBhgXcmJmv5eUV7x40J_ab2or"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

CLASS_LABELS = ["Lung_Opacity", "Normal", "Viral Pneumonia"]
IMG_SIZE = (224, 224)

# ── Global state ─────────────────────────────────────────────────────────────
model = None
model_lock = threading.Lock()


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True)
        print("Download complete.")
    else:
        print("Model already exists, skipping download.")


def initialize_model():
    global model
    download_model()
    print("Loading model into memory...")
    with model_lock:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lung Disease Classifier API",
    description="EfficientNetB0-based classifier for Lung Opacity, Normal, and Viral Pneumonia from chest X-rays.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize to 224×224 RGB and expand dims — no /255 normalization."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)                   # float32, 0-255
    img_array = tf.expand_dims(img_array, axis=0)                   # (1, 224, 224, 3)
    return img_array


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Lung Disease Classifier API is running.",
        "classes": CLASS_LABELS,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG or PNG image.",
        )

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try again shortly.")

    image_bytes = await file.read()

    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

    with model_lock:
        predictions = model.predict(img_array)

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_LABELS[predicted_index]
    confidence = float(predictions[0][predicted_index])

    all_confidences = {
        CLASS_LABELS[i]: float(predictions[0][i]) for i in range(len(CLASS_LABELS))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_confidences": {k: round(v, 4) for k, v in all_confidences.items()},
    }
