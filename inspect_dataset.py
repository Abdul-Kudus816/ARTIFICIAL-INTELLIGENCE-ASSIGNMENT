import os

# Path to our dataset
dataset_path = "dataset/standardized_256"

# Get all class folders
classes = sorted(os.listdir(dataset_path))

print("WASTE CLASSIFICATION DATASET")
print("=" * 40)

total_images = 0

for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    # Make sure it is a folder
    if os.path.isdir(class_path):

        # Get image files
        images = [
            file for file in os.listdir(class_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        number_of_images = len(images)
        total_images += number_of_images

        print(f"{class_name:15} : {number_of_images} images")

print("=" * 40)
print(f"Total images     : {total_images}")
print(f"Total classes    : {len(classes)}")