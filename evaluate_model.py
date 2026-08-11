import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

VAL_DIR = os.path.join(
    BASE_DIR,
    "DATASET",
    "validation"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "waste_classifier.keras"
)


# ============================================
# 2. SETTINGS
# ============================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


# ============================================
# 3. CHECK FILES
# ============================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found!")
    print(MODEL_PATH)
    exit()

if not os.path.exists(VAL_DIR):
    print("ERROR: Validation dataset not found!")
    print(VAL_DIR)
    exit()


# ============================================
# 4. LOAD MODEL
# ============================================

print("=" * 60)
print("AI WASTE CLASSIFICATION - MODEL EVALUATION")
print("=" * 60)

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================
# 5. LOAD VALIDATION DATA
# ============================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = validation_dataset.class_names

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ============================================
# 6. MODEL EVALUATION
# ============================================

print("\n")
print("=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print("\nOverall Results")
print("-" * 30)

print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy * 100:.2f}%")


# ============================================
# 7. MAKE PREDICTIONS
# ============================================

print("\nGenerating predictions...")

predictions = model.predict(
    validation_dataset,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)


# ============================================
# 8. GET TRUE LABELS
# ============================================

true_classes = np.concatenate([
    labels.numpy()
    for images, labels in validation_dataset
])


# ============================================
# 9. CLASSIFICATION REPORT
# ============================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names,
    digits=4
)

print(report)


# ============================================
# 10. CONFUSION MATRIX
# ============================================

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print(cm)


# ============================================
# 11. DISPLAY CONFUSION MATRIX
# ============================================

plt.figure(figsize=(12, 10))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("Waste Classification Confusion Matrix")

plt.tight_layout()

plt.show()


# ============================================
# 12. SUMMARY
# ============================================

print("\n")
print("=" * 60)
print("EVALUATION COMPLETE!")
print("=" * 60)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nModel location:")
print(MODEL_PATH)