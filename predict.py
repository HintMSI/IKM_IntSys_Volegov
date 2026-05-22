"""
  1. TTA (Test-Time Augmentation)
     Раньше модель получала ровно один кроп → одна ошибка решала всё.
     Теперь прогоняем несколько вариантов одного фото (центр, углы,
     зеркало, яркость) и УСРЕДНЯЕМ вероятности по всем вариантам.
     Это убирает "случайную" победу кабачка над огурцом.

  2. Мультиобъектное обнаружение (sliding window)
     Если на фото несколько объектов — делим изображение на окна,
     классифицируем каждое, фильтруем по уверенности и убираем
     дубликаты (non-maximum suppression по классу).

    python predict.py фото.jpg               # один объект
    python predict.py фото.jpg --multi        # несколько объектов
    python predict.py фото.jpg --top 5        # топ-5 кандидатов
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from PIL import Image, ImageEnhance

import tensorflow as tf

IMG_SIZE   = (100, 100)
SAVE_DIR   = "./saved_model"
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.keras")
NAMES_PATH = os.path.join(SAVE_DIR, "class_names.json")

# Минимальная уверенность чтобы считать находку валидной
CONFIDENCE_THRESHOLD = 0.35   # при мультиобъекте — порог на окно

# Цвета для рамок при мультиобъекте
BOX_COLORS = ["#e74c3c","#2ecc71","#3498db","#f39c12","#9b59b6","#1abc9c"]


# ЗАГРУЗКА МОДЕЛИ И КЛАССОВ
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}\n"
            "Сначала запустите train.py!"
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


# TTA — Test-Time Augmentation
# Суть: вместо одного предсказания делаем N,
# каждый раз чуть меняя фото, и усредняем.
# Это снижает влияние случайных артефактов
# (тень, угол, фон) на итоговый класс.

def make_tta_crops(img: Image.Image) -> list:
    """
    Возвращает список PIL-изображений для TTA:
      - оригинал по центру
      - 4 угловых кропа (75% от размера)
      - горизонтальное зеркало каждого из них
      - немного увеличенная яркость оригинала
      - немного уменьшенная яркость
    Итого: 12 вариантов -> усредняем вероятности.
    """
    w, h = img.size
    crops = []

    # Центральный кроп
    crops.append(img)

    # 4 угловых кропа (берём 80% по каждой стороне)
    crop_w, crop_h = int(w * 0.80), int(h * 0.80)
    corners = [
        (0,           0,            crop_w,       crop_h),       # верхний-левый
        (w - crop_w,  0,            w,             crop_h),       # верхний-правый
        (0,           h - crop_h,   crop_w,        h),            # нижний-левый
        (w - crop_w,  h - crop_h,   w,             h),            # нижний-правый
    ]
    for box in corners:
        crops.append(img.crop(box))

    # Зеркало каждого из 5 вариантов выше
    crops += [c.transpose(Image.FLIP_LEFT_RIGHT) for c in crops[:]]

    # Яркость ±20%
    crops.append(ImageEnhance.Brightness(img).enhance(1.2))
    crops.append(ImageEnhance.Brightness(img).enhance(0.8))

    return crops


def preprocess_crop(crop: Image.Image) -> np.ndarray:
    """
    Ресайз -> нормализация -> batch-размерность.
    """
    resized    = crop.resize(IMG_SIZE, Image.LANCZOS) #LANCZOS – качественная интерполяция(нахождение промежуточных знач. величины по известным знач. в соседних точка)
    arr        = np.array(resized, dtype=np.float32) / 255.0 #превращаем картинку в массив чисел
    return np.expand_dims(arr, axis=0)   # (1, 100, 100, 3) - новая размерность батча


def predict_with_tta(model, img: Image.Image) -> np.ndarray:
    """
    Запускает модель на всех TTA-кропах и возвращает
    усреднённый вектор вероятностей (shape: [N_CLASSES]).
    """
    crops  = make_tta_crops(img)
    probas = []

    # Собираем батч из всех кропов за один вызов модели
    batch  = np.concatenate([preprocess_crop(c) for c in crops], axis=0)
    preds  = model.predict(batch, verbose=0)   # (N_crops, N_classes)
    return preds.mean(axis=0)                  # усредняем по кропам


# ОДИНОЧНОЕ ПРЕДСКАЗАНИЕ
def predict_single(image_path: str, top_k: int = 5) -> dict:
    """Классифицирует одно изображение с TTA."""
    model, class_names = load_artifacts()
    img   = Image.open(image_path).convert("RGB")
    proba = predict_with_tta(model, img)

    top_idx     = np.argsort(proba)[::-1][:top_k] #тут получаем как раз топовые 5 индексов(вероятностей)
    top_results = [(class_names[i], float(proba[i])) for i in top_idx]

    return {
        "mode":          "single",
        "predicted":     top_results[0][0],
        "confidence":    top_results[0][1],
        "top":           top_results,
        "img":           img,
        "original_size": img.size,
    }


# МУЛЬТИОБЪЕКТНОЕ ОБНАРУЖЕНИЕ — sliding window (пока недоделано)
#   1. Делим фото на перекрывающиеся окна
#   2. Классифицируем каждое окно с TTA
#   3. Оставляем только уверенные предсказания
#   4. Убираем дублирующиеся классы (один класс → одна лучшая рамка)

def predict_multi(image_path: str, top_k: int = 5) -> dict:
    """
    Ищет несколько объектов на одном фото.
    
    Параметры окон подобраны для фото среднего размера.
    Если объекты очень маленькие — уменьшите WINDOW_SCALES.
    """
    model, class_names = load_artifacts()
    img  = Image.open(image_path).convert("RGB")
    W, H = img.size

    # Масштабы окон: берём 100%, 70%, 45% от размера фото
    WINDOW_SCALES = [1.0, 0.70, 0.45]
    STRIDE_RATIO  = 0.50   # шаг = 50% от размера окна (перекрытие 50%)

    candidates = []   # (class, confidence, box=(x1,y1,x2,y2))

    for scale in WINDOW_SCALES:
        win_w = max(int(W * scale), IMG_SIZE[0])
        win_h = max(int(H * scale), IMG_SIZE[1])
        step_x = max(int(win_w * STRIDE_RATIO), 1)
        step_y = max(int(win_h * STRIDE_RATIO), 1)

        for y in range(0, H - win_h + 1, step_y):
            for x in range(0, W - win_w + 1, step_x):
                box  = (x, y, x + win_w, y + win_h)
                crop = img.crop(box)

                # TTA на каждом окне — дороже, но точнее
                proba     = predict_with_tta(model, crop)
                best_idx  = int(np.argmax(proba))
                best_conf = float(proba[best_idx])

                if best_conf >= CONFIDENCE_THRESHOLD:
                    candidates.append({
                        "class":      class_names[best_idx],
                        "confidence": best_conf,
                        "box":        box,
                    })

    # Добавляем полное фото как кандидата
    proba     = predict_with_tta(model, img)
    best_idx  = int(np.argmax(proba))
    candidates.append({
        "class":      class_names[best_idx],
        "confidence": float(proba[best_idx]),
        "box":        (0, 0, W, H),
        "top":        [(class_names[i], float(proba[i]))
                       for i in np.argsort(proba)[::-1][:top_k]],
    })

    # NMS по классу: для каждого уникального класса — берём окно с max уверенностью
    seen_classes = {}
    for c in sorted(candidates, key=lambda x: -x["confidence"]):
        cls = c["class"]
        if cls not in seen_classes:
            seen_classes[cls] = c

    detections = list(seen_classes.values())
    # Сортируем по уверенности
    detections.sort(key=lambda x: -x["confidence"])

    return {
        "mode":       "multi",
        "detections": detections,
        "img":        img,
        "original_size": img.size,
    }


# ВИЗУАЛИЗАЦИЯ
def visualize_single(result: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fruit Classifier (MobileNetV2 + TTA)", fontsize=13, fontweight="bold")

    axes[0].imshow(result["img"])
    axes[0].set_title(
        f"Фото: {result['original_size'][0]}×{result['original_size'][1]} px", fontsize=10
    )
    axes[0].axis("off")

    names  = [r[0] for r in result["top"]]
    probs  = [r[1] * 100 for r in result["top"]]
    colors = ["#2ecc71" if i == 0 else "#95a5a6" for i in range(len(names))]

    bars = axes[1].barh(names[::-1], probs[::-1], color=colors[::-1], edgecolor="white")
    axes[1].set_xlim(0, 115)
    axes[1].set_xlabel("Вероятность (%)")
    axes[1].set_title(f"Топ-{len(names)} (TTA усреднение)")
    for bar, p in zip(bars, probs[::-1]):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{p:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150)
    plt.show()


def visualize_multi(result: dict):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(result["img"])
    ax.set_title("Обнаруженные объекты (sliding window + TTA)", fontsize=12, fontweight="bold")
    ax.axis("off")

    for i, det in enumerate(result["detections"]):
        x1, y1, x2, y2 = det["box"]
        color = BOX_COLORS[i % len(BOX_COLORS)]
        rect  = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        label = f"{det['class']}  {det['confidence']*100:.0f}%"
        ax.text(
            x1 + 4, y1 - 6, label,
            color="white", fontsize=9, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground=color)]
        )

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Классификатор фруктов/овощей/орехов")
    parser.add_argument("image",   help="Путь к изображению")
    parser.add_argument("--top",   type=int, default=5, help="Топ-N предсказаний (одиночный режим)")
    parser.add_argument("--multi", action="store_true",  help="Режим нескольких объектов")
    parser.add_argument("--no-plot", action="store_true", help="Без графика")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f" Файл не найден: {args.image}")
        sys.exit(1)

    print(f"  {args.image}")
    print(f" Режим: {'мультиобъект' if args.multi else 'одиночный + TTA'}\n")

    if args.multi:
        result = predict_multi(args.image, top_k=args.top)
        print(f"Найдено объектов: {len(result['detections'])}\n")
        for i, d in enumerate(result["detections"], 1):
            print(f"  {i}. {d['class']:<35} {d['confidence']*100:5.1f}%")
        if not args.no_plot:
            visualize_multi(result)
    else:
        result = predict_single(args.image, top_k=args.top)
        print(f"  {result['predicted']}  ({result['confidence']*100:.1f}%)\n")
        print(f"Топ-{args.top}:")
        for i, (cls, prob) in enumerate(result["top"]):
            bar = "█" * int(prob * 30)
            print(f"  {i+1}. {cls:<35} {prob*100:5.1f}%  {bar}")
        if not args.no_plot:
            visualize_single(result)
