import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "DATASET", "train")
VAL_DIR = os.path.join(BASE_DIR, "DATASET", "validation")

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "waste_classifier.keras"
)


# ============================================
# 2. SETTINGS
# ============================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

SEED = 42


# ============================================
# 3. CHECK DATASET
# ============================================

if not os.path.exists(TRAIN_DIR):
    print("ERROR: Training dataset not found!")
    print(TRAIN_DIR)
    exit()

if not os.path.exists(VAL_DIR):
    print("ERROR: Validation dataset not found!")
    print(VAL_DIR)
    exit()


print("=" * 60)
print("AI WASTE CLASSIFICATION - MODEL TRAINING")
print("=" * 60)

print("\nTraining directory:")
print(TRAIN_DIR)

print("\nValidation directory:")
print(VAL_DIR)


# ============================================
# 4. LOAD TRAINING DATA
# ============================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================
# 5. LOAD VALIDATION DATA
# ============================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================
# 6. GET CLASS NAMES
# ============================================

class_names = train_dataset.class_names

NUM_CLASSES = len(class_names)

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print("\nNumber of classes:", NUM_CLASSES)


# ============================================
# 7. IMPROVE DATA PIPELINE PERFORMANCE
# ============================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================
# 8. DATA AUGMENTATION
# ============================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])


# ============================================
# 9. BUILD CNN MODEL
# ============================================

model = models.Sequential([

    # Data augmentation
    data_augmentation,

    # Normalize pixels from 0-255 to 0-1
    layers.Rescaling(
        1.0 / 255
    ),

    # First convolution block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Third convolution block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Fourth convolution block
    layers.Conv2D(
        256,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convert feature maps to vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        256,
        activation="relu"
    ),

    # Prevent overfitting
    layers.Dropout(0.5),

    # Output layer
    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================
# 10. COMPILE MODEL
# ============================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================
# 11. DISPLAY MODEL STRUCTURE
# ============================================

print("\n")
model.summary()


# ============================================
# 12. CALLBACKS
# ============================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# ============================================
# 13. TRAIN MODEL
# ============================================

print("\n")
print("=" * 60)
print("STARTING MODEL TRAINING")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ============================================
# 14. SAVE FINAL MODEL
# ============================================

model.save(MODEL_PATH)

print("\n")
print("=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nBest validation accuracy:")
print(
    f"{max(history.history['val_accuracy']) * 100:.2f}%"
)