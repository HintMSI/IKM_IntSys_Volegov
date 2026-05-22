#transfer-learning
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score
)

DATASET_ROOT = "./fruits-360"

IMG_SIZE     = (100, 100)              # размер входного изображения
BATCH_SIZE   = 32
EPOCHS_FREEZE = 10                     # эпохи с замороженной предобученной частью
EPOCHS_FINE   = 10                     # эпохи уже с размороженной предобуч. частью
SEED         = 42
SAVE_DIR     = "./saved_model"
RESULTS_DIR  = "./results"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU доступен: {bool(tf.config.list_physical_devices('GPU'))}")


# 1. ЗАГРУЗКА ДАННЫХ С АУГМЕНТАЦИЕЙ (аналог ColumnTransformer из лабы)
train_dir = os.path.join(DATASET_ROOT, "Training")
test_dir  = os.path.join(DATASET_ROOT, "Test")

# Аугментация(создания искусственных примеров из имеющихся)
# только для обучающей выборки — расширяем датасет искусственно
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # нормализация пикселей [0,1] — как StandardScaler
    rotation_range=15,          # случайный поворот ±15°
    width_shift_range=0.1,      # горизонтальный сдвиг
    height_shift_range=0.1,     # вертикальный сдвиг
    zoom_range=0.1,             # случайный зум
    horizontal_flip=True,       # зеркальное отражение
    validation_split=0.15       # 15% train на проверку,чтоб не было переобучения
)

# Для теста — только нормализация, без аугментации
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

#В папке training создает метки классов по папкам(создает батчи
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
    shuffle=True #чтоб модель не запомнила порядок,то перемешиваем между эпохами
)

#тут валидационный генератор без перемешивания для оценки результата
val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
    shuffle=False
)
# и генератор для тестовых данных
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

#узнаем сколько всего получилось классов и их названий
NUM_CLASSES = train_gen.num_classes
CLASS_NAMES = list(train_gen.class_indices.keys())

print(f"\nКлассов: {NUM_CLASSES}")
print(f"Train батчей: {len(train_gen)}, Val батчей: {len(val_gen)}, Test батчей: {len(test_gen)}")

# Сохраняем маппинг классов для predict.py,чтобы потом он понимал какие мои картинки соотносить к каким конкретно классам
with open(os.path.join(SAVE_DIR, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)


# 2. ПОСТРОЕНИЕ МОДЕЛИ — Transfer Learning
# Аналог: берём готовый Pipeline и тюним его, тк MobileNetV2 уже обучена на 1.2кк изображений
def build_model(num_classes: int) -> tf.keras.Model:
    # Базовая модель — предобученные веса ImageNet
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3), #картинки 100 на 100 с 3 цветовыми каналами
        include_top=False,         # убираем голову ImageNet (1000 классов),тк у нас свое чилсо классов
        weights="imagenet"
    )
    base_model.trainable = False   # замораживаем все ее слои,тк пока будем учить новые слои,которые мы будем добавлять

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3)) #точка входа картинки
    x = base_model(inputs, training=False) # пропускаем через замороженную модель, она будет выделять признаки
    x = layers.GlobalAveragePooling2D()(x)   # усредняем карты признаков от переобучения(двумерную карту преобразуют в один вектор)
    x = layers.BatchNormalization()(x)  #нормализуем эти активации для стабильности обучения
    x = layers.Dense(256, activation="relu")(x) #полносвязный слой на 256 нейронов с активацией выпрямителя relu(отриц знач - 0, полож. как есть)
    x = layers.Dropout(0.4)(x)               # Dropout — как Ridge: борьба с переобучением( выключение 40% случайных нейронов на каждом шаге от переобучения)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x) #дает вероятности для каждого класса

    model = models.Model(inputs, outputs)
    return model, base_model #возвращаем базовую модель и базовую часть,чтоб потом разморозить


model, base_model = build_model(NUM_CLASSES)
model.summary()


# 3. ФАЗА 1 — обучаем только голову
# (base_model заморожена — учим только наши Dense-слои)

model.compile(
    #learning_rate=1e-3 – скорость обучения 0.001 (шаг, на который обновляются веса).
    optimizer=optimizers.Adam(learning_rate=1e-3), #оптимизатор ,который обновляет веса для классов
    loss="categorical_crossentropy",  #функция потерь для штрафа модели
    metrics=["accuracy"]
)

cb_list = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True), # если accur не растет 5 эпох подряд,то все останавливается и воостанавливаются лучшие веса от переобучения
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6), #если потери на валидации перестали уменьшаться,то уменьшаем шаг обучения в 2 раза
    callbacks.ModelCheckpoint(  # сохраняет лучшую модель по accur
        os.path.join(SAVE_DIR, "best_model.keras"),
        monitor="val_accuracy", save_best_only=True
    )
]

print("ФАЗА 1: Обучение головы (base заморожена)")
#сам запуск обучения
history1 = model.fit(
    train_gen, #бесконечно генерит пакеты с аугментированными картинками
    validation_data=val_gen, #проверочная выборка
    epochs=EPOCHS_FREEZE,
    callbacks=cb_list,
    verbose=1
)


# 4. ФАЗ 2 — Fine-tuning (размораживаем верхние слои предобученной части) Аналог: GridSearchCV с fine-tuning гиперпараметров
print("ФАЗ 2: Fine-tuning (размораживаем последние 30 слоёв)")

base_model.trainable = True
# Размораживаем только верхние 30 слоёв ,тк они отвечают за абстрактные признаки ,а не за простые геом. формы
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Очень маленький lr(уменьшили в 10 раз) при fine-tuning — чтобы не «сломать» ImageNet-веса
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

#второй этап обучения (аналогичен первоому)
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=cb_list,
    verbose=1
)


# 5. ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ
print("="*67)
print("ФИНАЛЬНАЯ ОЦЕНКА НА TEST SET")


# Загружаем лучшую сохранённую модель
best_model = tf.keras.models.load_model(os.path.join(SAVE_DIR, "best_model.keras"))

test_loss, test_acc = best_model.evaluate(test_gen, verbose=0)
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Loss:     {test_loss:.4f}")

# Предсказания для детального отчёта
print("\nВычисляем предсказания...")
y_pred_proba = best_model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = test_gen.classes

# Classification Report
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
print("\n--- Classification Report (macro avg) ---")
print(f"Precision: {report['macro avg']['precision']:.4f}")
print(f"Recall:    {report['macro avg']['recall']:.4f}")
print(f"F1-score:  {report['macro avg']['f1-score']:.4f}")

# Сохраняем полный отчёт
full_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write(full_report)
print(f"\nОтчёт сохранён в {RESULTS_DIR}/classification_report.txt")



# 6. ГРАФИКИ

# 6.1 Кривые обучения
def merge_histories(h1, h2):
    merged = {}
    for k in h1.history:
        merged[k] = h1.history[k] + h2.history[k]
    return merged

hist = merge_histories(history1, history2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Кривые обучения — Fruit Classifier (MobileNetV2)", fontsize=14)

axes[0].plot(hist["accuracy"],     label="Train Accuracy", color="royalblue")
axes[0].plot(hist["val_accuracy"], label="Val Accuracy",   color="orange", linestyle="--")
axes[0].axvline(EPOCHS_FREEZE - 1, color="gray", linestyle=":", label="Fine-tune начало")
axes[0].set_title("Accuracy"); axes[0].set_xlabel("Эпоха"); axes[0].legend()

axes[1].plot(hist["loss"],     label="Train Loss", color="royalblue")
axes[1].plot(hist["val_loss"], label="Val Loss",   color="orange", linestyle="--")
axes[1].axvline(EPOCHS_FREEZE - 1, color="gray", linestyle=":", label="Fine-tune начало")
axes[1].set_title("Loss"); axes[1].set_xlabel("Эпоха"); axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"), dpi=150)
plt.show()
print("График сохранён: learning_curves.png")


# 6.2 Матрица ошибок — топ-20 классов (полная будет нечитаема)
print("\nСтроим матрицу ошибок (топ-20 классов по ошибкам)...")

cm = confusion_matrix(y_true, y_pred)
# Берём 20 классов с наибольшим числом ошибок
errors_per_class = cm.sum(axis=1) - np.diag(cm)
top20_idx = np.argsort(errors_per_class)[-20:]
cm_top20  = cm[np.ix_(top20_idx, top20_idx)]
names_top20 = [CLASS_NAMES[i] for i in top20_idx]

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm_top20, annot=True, fmt="d", cmap="Blues",
            xticklabels=names_top20, yticklabels=names_top20, ax=ax)
ax.set_title("Матрица ошибок — топ-20 проблемных классов", fontsize=13)
ax.set_xlabel("Предсказано"); ax.set_ylabel("Истинно")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_top20.png"), dpi=150)
plt.show()
print("График сохранён: confusion_matrix_top20.png")


# 6.3 Топ-10 худших классов (по F1) — таблица
print("\n--- Топ-10 классов с наименьшим F1-score ---")
class_f1 = {cls: report[cls]["f1-score"] for cls in CLASS_NAMES if cls in report}
worst_10 = sorted(class_f1.items(), key=lambda x: x[1])[:10]
for cls, f1 in worst_10:
    support = int(report[cls]["support"])
    print(f"  {cls:<30} F1={f1:.3f}  (n={support})")


# 6.4 Итоговая таблица
print("\n" + "="*55)
print("ИТОГОВАЯ ТАБЛИЦА МЕТРИК")
print("="*55)
print(f"{'Метрика':<25} {'Значение':>10}")
print("-"*36)
print(f"{'Test Accuracy':<25} {test_acc*100:>9.2f}%")
print(f"{'Macro Precision':<25} {report['macro avg']['precision']:>10.4f}")
print(f"{'Macro Recall':<25} {report['macro avg']['recall']:>10.4f}")
print(f"{'Macro F1-score':<25} {report['macro avg']['f1-score']:>10.4f}")
print(f"{'Weighted F1-score':<25} {report['weighted avg']['f1-score']:>10.4f}")
print("="*55)

print(f"\nМодель сохранена: {SAVE_DIR}/best_model.keras")
print(f"Результаты сохранены в папку: {RESULTS_DIR}/")
