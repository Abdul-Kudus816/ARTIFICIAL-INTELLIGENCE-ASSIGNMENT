import os
import shutil
import random

# ==============================
# SETTINGS
# ==============================

# Location of the original dataset
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "DATASET",
    "standardized_256"
)

# Location for the prepared dataset
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "DATASET"
)

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "validation")

# 80% training, 20% validation
TRAIN_RATIO = 0.80

# Make the split reproducible
random.seed(42)


# ==============================
# CHECK SOURCE DATASET
# ==============================

if not os.path.exists(SOURCE_DIR):
    print("ERROR: Source dataset not found!")
    print("Looking for:")
    print(SOURCE_DIR)
    exit()

print("WASTE CLASSIFICATION DATASET PREPARATION")
print("=" * 50)

print("Source dataset:")
print(SOURCE_DIR)

print()


# ==============================
# GET CLASSES
# ==============================

classes = sorted([
    folder
    for folder in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, folder))
])

print("Classes found:")
for class_name in classes:
    print("-", class_name)

print()
print("Total classes:", len(classes))
print()


# ==============================
# CREATE OUTPUT DIRECTORIES
# ==============================

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)


# ==============================
# PROCESS EACH CLASS
# ==============================

total_images = 0
total_train = 0
total_validation = 0

for class_name in classes:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    train_class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    val_class_dir = os.path.join(
        VAL_DIR,
        class_name
    )

    # Create class folders
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Get image files
    images = [
        file
        for file in os.listdir(source_class_dir)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    # Shuffle images
    random.shuffle(images)

    # Calculate training size
    train_count = int(len(images) * TRAIN_RATIO)

    train_images = images[:train_count]
    validation_images = images[train_count:]

    # Copy training images
    for image in train_images:

        source = os.path.join(
            source_class_dir,
            image
        )

        destination = os.path.join(
            train_class_dir,
            image
        )

        shutil.copy2(source, destination)

    # Copy validation images
    for image in validation_images:

        source = os.path.join(
            source_class_dir,
            image
        )

        destination = os.path.join(
            val_class_dir,
            image
        )

        shutil.copy2(source, destination)

    # Update totals
    total_images += len(images)
    total_train += len(train_images)
    total_validation += len(validation_images)

    print(
        f"{class_name:<15} : "
        f"{len(images):>5} total | "
        f"{len(train_images):>5} train | "
        f"{len(validation_images):>5} validation"
    )


# ==============================
# FINAL SUMMARY
# ==============================

print("=" * 50)

print("DATASET PREPARATION COMPLETE!")
print()

print("Total images      :", total_images)
print("Training images   :", total_train)
print("Validation images :", total_validation)

print()

print("Training folder:")
print(TRAIN_DIR)

print()

print("Validation folder:")
print(VAL_DIR)

print("=" * 50)