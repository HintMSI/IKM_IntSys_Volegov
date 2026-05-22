"""
  predict2.py — Предсказание для ЛЮБОГО изображения
  
  Использование:
    python predict2.py путь/к/фото.jpg
    python predict2.py путь/к/фото.jpg --top 5

"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import tensorflow as tf
from PIL import Image



IMG_SIZE   = (100, 100)
SAVE_DIR   = "./saved_model"
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.keras")
NAMES_PATH = os.path.join(SAVE_DIR, "class_names.json")


def load_artifacts():
    """Загружает модель и список классов."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}\n"
            "Сначала запустите train.py!"
        )
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)

    return model, class_names


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Загружает изображение любого размера и готовит его для модели.
    
    Шаги:
      1. Открыть файл (JPEG, PNG, BMP, WEBP — PIL справится)
      2. Конвертировать в RGB (на случай RGBA / grayscale)
      3. Ресайзить до 100×100 (LANCZOS — высокое качество)
      4. Нормализовать пиксели в [0, 1] — как StandardScaler в лабах
      5. Добавить batch-размерность: (100,100,3) → (1,100,100,3)
    """
    img = Image.open(image_path).convert("RGB")
    original_size = img.size         # сохраним для вывода
    img_resized = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, img, original_size


def predict(image_path: str, top_k: int = 5) -> dict:
    """
    Основная функция предсказания.
    
    Возвращает словарь с:
      - top_classes: список (класс, вероятность)
      - predicted_class: строка
      - confidence: float
    """
    model, class_names = load_artifacts()

    img_batch, img_pil, original_size = preprocess_image(image_path)

    # Предсказание — вектор вероятностей по всем классам
    proba = model.predict(img_batch, verbose=0)[0]   # shape: (N_CLASSES,)

    # Топ-K классов (как argmax, но для нескольких)
    top_indices = np.argsort(proba)[::-1][:top_k]
    top_results = [(class_names[i], float(proba[i])) for i in top_indices]

    return {
        "predicted_class": top_results[0][0],
        "confidence":      top_results[0][1],
        "top_classes":     top_results,
        "original_size":   original_size,
        "img_pil":         img_pil
    }


def visualize_prediction(result: dict, image_path: str):
    """Красивая визуализация результата."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fruit Classifier — результат предсказания", fontsize=14, fontweight="bold")

    # Левая панель — исходное фото
    axes[0].imshow(result["img_pil"])
    axes[0].set_title(
        f"Входное изображение\n{result['original_size'][0]}×{result['original_size'][1]} px",
        fontsize=11
    )
    axes[0].axis("off")

    # Цветная рамка по уверенности
    conf = result["confidence"]
    color = "green" if conf > 0.8 else ("orange" if conf > 0.5 else "red")
    for spine in axes[0].spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(4)

    # Правая панель — Bar chart вероятностей (топ-K)
    names  = [r[0] for r in result["top_classes"]]
    probs  = [r[1] * 100 for r in result["top_classes"]]
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(names))]

    bars = axes[1].barh(names[::-1], probs[::-1], color=colors[::-1], edgecolor="white")
    axes[1].set_xlabel("Вероятность (%)", fontsize=11)
    axes[1].set_title(f"Топ-{len(names)} предсказаний", fontsize=11)
    axes[1].set_xlim(0, 110)

    for bar, p in zip(bars, probs[::-1]):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{p:.1f}%", va="center", fontsize=10, fontweight="bold")

    patch_pred = mpatches.Patch(color="#2ecc71", label="Победитель")
    patch_rest = mpatches.Patch(color="#3498db", label="Остальные кандидаты")
    axes[1].legend(handles=[patch_pred, patch_rest], loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path = "prediction_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Визуализация сохранена: {out_path}")



# ЗАПУСК ИЗ КОМАНДНОЙ СТРОКИ
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Классификатор фруктов/овощей/орехов")
    parser.add_argument("image", help="Путь к изображению (любой размер)")
    parser.add_argument("--top", type=int, default=5, help="Показать топ-N предсказаний (по умолчанию 5)")
    parser.add_argument("--no-plot", action="store_true", help="Без графика")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f" Файл не найден: {args.image}")
        sys.exit(1)

    print(f" Изображение: {args.image}")
    print(" Распознаём...")

    result = predict(args.image, top_k=args.top)

    print(f"\n Результат:")
    print(f"   Класс:       {result['predicted_class']}")
    print(f"   Уверенность: {result['confidence']*100:.1f}%")
    print(f"   Исходный размер: {result['original_size'][0]}×{result['original_size'][1]} px")
    print(f"   (ресайзнуто до {IMG_SIZE[0]}×{IMG_SIZE[1]} для модели)")

    print(f"\n📊 Топ-{args.top} кандидатов:")
    for i, (cls, prob) in enumerate(result["top_classes"]):
        bar = "█" * int(prob * 30)
        marker = " ← 🏆" if i == 0 else ""
        print(f"   {i+1}. {cls:<30} {prob*100:5.1f}%  {bar}{marker}")

    if not args.no_plot:
        visualize_prediction(result, args.image)
