import os
import matplotlib.pyplot as plt
from PIL import Image

# Path to the actual dataset
dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "DATASET",
    "standardized_256"
)

# Check if dataset exists
if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    print("Looking for:", dataset_path)
    exit()

# Get the waste categories
classes = sorted([
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
])

print("Classes found:")
for class_name in classes:
    print("-", class_name)

print("\nTotal classes:", len(classes))

# Create figure
plt.figure(figsize=(15, 10))

plot_number = 1

# Display one image from each class
for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    # Get image files
    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    # Skip empty folders
    if len(images) == 0:
        print(f"No images found in {class_name}")
        continue

    # Select the first image
    image_path = os.path.join(class_path, images[0])

    # Open image
    image = Image.open(image_path)

    # Display image
    plt.subplot(2, 5, plot_number)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis("off")

    plot_number += 1

plt.suptitle("Waste Classification Dataset Samples", fontsize=16)
plt.tight_layout()
plt.show()