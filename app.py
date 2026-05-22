"""
=============================================================
  app.py — Flask веб-приложение для Fruit Classifier
  
  Маршруты:
    GET  /          → главная страница
    POST /predict   → JSON: одиночный или мульти режим
    
  Запуск:
    python app.py
    Открыть: http://localhost:5000
=============================================================
"""

import io
import json
import base64
import traceback
import numpy as np
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from PIL import Image

# Импортируем логику из predict.py
from predict import load_artifacts, predict_with_tta, predict_multi, IMG_SIZE, CONFIDENCE_THRESHOLD

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB максимум

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}

# ── Модель загружается один раз при старте сервера ────────
# (а не заново при каждом запросе — так намного быстрее)
MODEL = None
CLASS_NAMES = None

def ensure_model():
    global MODEL, CLASS_NAMES
    if MODEL is None:
        MODEL, CLASS_NAMES = load_artifacts()


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def img_to_b64(img: Image.Image, max_size: int = 500) -> str:
    """Сжимаем превью и конвертируем в base64 для браузера."""
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────
# МАРШРУТЫ
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Принимает: multipart/form-data
      - image: файл изображения
      - mode: "single" (по умолчанию) или "multi"
      
    Возвращает JSON:
      single → {success, mode, predicted, confidence, top: [{class, confidence}], preview_b64}
      multi  → {success, mode, detections: [{class, confidence, box}], preview_b64}
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "Файл не найден"}), 400

    file = request.files["image"]
    mode = request.form.get("mode", "single")   # "single" или "multi"

    if not file.filename or not allowed(file.filename):
        return jsonify({"success": False, "error": "Недопустимый формат файла"}), 400

    try:
        ensure_model()
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        preview_b64 = img_to_b64(img.copy())

        if mode == "multi":
            # Мультиобъектный режим — sliding window + TTA на каждом окне
            W, H = img.size
            WINDOW_SCALES = [1.0, 0.70, 0.45]
            STRIDE_RATIO  = 0.50
            candidates = []

            for scale in WINDOW_SCALES:
                win_w = max(int(W * scale), IMG_SIZE[0])
                win_h = max(int(H * scale), IMG_SIZE[1])
                step_x = max(int(win_w * STRIDE_RATIO), 1)
                step_y = max(int(win_h * STRIDE_RATIO), 1)

                for y in range(0, H - win_h + 1, step_y):
                    for x in range(0, W - win_w + 1, step_x):
                        box  = (x, y, x + win_w, y + win_h)
                        crop = img.crop(box)
                        proba    = predict_with_tta(MODEL, crop)
                        best_idx = int(np.argmax(proba))
                        best_conf = float(proba[best_idx])
                        if best_conf >= CONFIDENCE_THRESHOLD:
                            candidates.append({
                                "class":      CLASS_NAMES[best_idx],
                                "confidence": round(best_conf * 100, 1),
                                "box":        list(box),
                            })

            # Полное фото как базовый кандидат
            proba    = predict_with_tta(MODEL, img)
            best_idx = int(np.argmax(proba))
            candidates.append({
                "class":      CLASS_NAMES[best_idx],
                "confidence": round(float(proba[best_idx]) * 100, 1),
                "box":        [0, 0, W, H],
            })

            # NMS: оставляем лучшее окно на каждый уникальный класс
            seen = {}
            for c in sorted(candidates, key=lambda x: -x["confidence"]):
                if c["class"] not in seen:
                    seen[c["class"]] = c
            detections = sorted(seen.values(), key=lambda x: -x["confidence"])

            return jsonify({
                "success":    True,
                "mode":       "multi",
                "detections": detections,
                "preview_b64": preview_b64,
                "img_size":   [W, H],
            })

        else:
            # Одиночный режим с TTA
            proba    = predict_with_tta(MODEL, img)
            top_k    = 5
            top_idx  = np.argsort(proba)[::-1][:top_k]
            top      = [{"class": CLASS_NAMES[i], "confidence": round(float(proba[i]) * 100, 1)}
                        for i in top_idx]

            return jsonify({
                "success":     True,
                "mode":        "single",
                "predicted":   top[0]["class"],
                "confidence":  top[0]["confidence"],
                "top":         top,
                "preview_b64": preview_b64,
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    ensure_model()
    print("\n🍎  Fruit Classifier  →  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
