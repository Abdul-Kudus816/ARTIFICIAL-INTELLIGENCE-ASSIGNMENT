import os
import matplotlib.pyplot as plt

# Path to the dataset
dataset_path = "dataset/standardized_256"

# Get the class names
classes = sorted(os.listdir(dataset_path))

# Count images in each class
image_counts = []

for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):

        count = len([
            file for file in os.listdir(class_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        image_counts.append(count)

# Display the results
print("Image Distribution")
print("=" * 40)

for class_name, count in zip(classes, image_counts):
    print(f"{class_name:15} : {count}")

# Create the bar chart
plt.figure(figsize=(10, 6))

plt.bar(classes, image_counts)

plt.title("Waste Dataset Class Distribution")
plt.xlabel("Waste Category")
plt.ylabel("Number of Images")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()