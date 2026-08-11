# ARTIFICIAL INTELLIGENCE (MACHINE LEARNING) PROJECT REPORT

## AI WASTE CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORK

## GROUP MEMBERS

| NameIndex NumberGitHub Repository |            |                                                                      |
| --------------------------------- | ---------- | -------------------------------------------------------------------- |
| Frederick Gyabeng                 | UEB3505123 | https://github.com/freakynet/AI_WASTE_CLASSIFICATION                 |
| Nkrumah Philimon                  | UEB3510823 | https://github.com/P798/AI_WAIST_-CLASSIFICATION_ASSIGNMENT          |
| Yahaya Abdul-Kudus                | UEB3515723 | https://github.com/Abdul-Kudus816/ARTIFICIAL-INTELLIGENCE-ASSIGNMENT |


# ABSTRACT

This project focuses on the development of an Artificial Intelligence system for classifying different types of waste using images. A Convolutional Neural Network (CNN) was developed using Python, TensorFlow, and Keras.

The dataset contained 12,259 images belonging to 10 waste categories: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, and trash.

The dataset was divided into 9,802 training images and 2,457 validation images. The images were resized to 128 × 128 pixels before being used to train the model.

The developed CNN achieved a validation accuracy of 73.30%. The results show that Artificial Intelligence can be used to automatically classify different types of waste from images.


# 1. INTRODUCTION

Waste classification is an important part of waste management and recycling. Different types of waste require different methods of disposal or recycling. However, manually sorting waste can be time-consuming and may result in errors.

Artificial Intelligence can be used to automate this process. In this project, a Convolutional Neural Network was developed to identify and classify waste images into ten different categories.

CNNs are commonly used for image classification because they can automatically learn important visual features from images.


# 2. PROJECT OBJECTIVES

The main objectives of the project were to:

* Obtain a waste image dataset.
* Inspect and prepare the dataset.
* Divide the dataset into training and validation sets.
* Develop a CNN model for image classification.
* Train the model using the waste images.
* Evaluate the performance of the trained model.
* Determine the accuracy of the model.
* Analyze the classification performance of each waste category.


# 3. DATASET

The dataset used for this project contained 12,259 images divided into ten waste categories.

### Dataset Distribution

| Waste CategoryNumber of Images |        |
| ------------------------------ | ------ |
| Battery                        | 756    |
| Biological                     | 699    |
| Cardboard                      | 1,411  |
| Clothes                        | 1,892  |
| Glass                          | 1,736  |
| Metal                          | 930    |
| Paper                          | 1,336  |
| Plastic                        | 1,597  |
| Shoes                          | 1,449  |
| Trash                          | 453    |
| Total                          | 12,259 |

The dataset was divided into:

* Training dataset: 9,802 images
* Validation dataset: 2,457 images

This represents approximately an 80:20 training-to-validation split.


# 4. METHODOLOGY

The project followed these major steps:

Dataset
   ↓
Dataset Inspection
   ↓
Dataset Preparation
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
CNN Model Development
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Classification Results

## 4.1 Image Preprocessing

The images were resized to:

128 × 128 pixels

The pixel values were normalized from the range 0–255 to the range 0–1 using:

1 / 255


Data augmentation was also applied to increase the variety of training images. The techniques used were:

* Random horizontal flipping
* Random rotation
* Random zoom
* Random contrast adjustment

# 5. CNN MODEL

A Convolutional Neural Network was developed using TensorFlow and Keras.

The model contained four convolutional layers with:

* 32 filters
* 64 filters
* 128 filters
* 256 filters

Each convolutional layer used the ReLU activation function and was followed by a max-pooling layer.

The extracted features were then passed through:

* Flatten layer
* Dense layer with 256 neurons
* Dropout layer with a rate of 0.5
* Output layer with 10 neurons

The output layer used the Softmax activation function to produce the probability of each waste category.

### CNN Structure

Input Image
128 × 128
      ↓
Data Augmentation
      ↓
Rescaling
      ↓
Conv2D - 32
      ↓
Max Pooling
      ↓
Conv2D - 64
      ↓
Max Pooling
      ↓
Conv2D - 128
      ↓
Max Pooling
      ↓
Conv2D - 256
      ↓
Max Pooling
      ↓
Flatten
      ↓
Dense - 256
      ↓
Dropout - 0.5
      ↓
Softmax
      ↓
10 Waste Classes
```

# 6. MODEL TRAINING

The model was trained using the Adam optimizer with a learning rate of 0.001.

The loss function used was:

Sparse Categorical Crossentropy

The training configuration included:

| ParameterValue    |                                 |
| ----------------- | ------------------------------- |
| Image Size        | 128 × 128                       |
| Batch Size        | 32                              |
| Maximum Epochs    | 30                              |
| Optimizer         | Adam                            |
| Learning Rate     | 0.001                           |
| Loss Function     | Sparse Categorical Crossentropy |
| Number of Classes | 10                              |
| Dropout           | 0.5                             |

Early stopping was used with a patience of 5 epochs. Model checkpointing was also used to save the model with the best validation accuracy.

The final trained model was saved as:

waste_classifier.keras


# 7. RESULTS

After training, the model was evaluated using the validation dataset.

### Overall Results

| MetricResult        |        |
| ------------------- | ------ |
| Validation Accuracy | 73.30% |
| Validation Loss     | 0.8970 |
| Macro F1-Score      | 71.68% |
| Weighted F1-Score   | 73.27% |

The model achieved an overall validation accuracy of 73.30%.

This means that the model correctly classified approximately 73 out of every 100 validation images.

# 8. CLASSIFICATION PERFORMANCE

| ClassPrecisionRecallF1-Score |        |        |        |
| ---------------------------- | ------ | ------ | ------ |
| Battery                      | 84.33% | 74.34% | 79.02% |
| Biological                   | 77.60% | 69.29% | 73.21% |
| Cardboard                    | 78.75% | 75.97% | 77.34% |
| Clothes                      | 88.16% | 88.39% | 88.27% |
| Glass                        | 73.04% | 80.17% | 76.44% |
| Metal                        | 57.53% | 57.53% | 57.53% |
| Paper                        | 59.30% | 63.06% | 61.12% |
| Plastic                      | 66.36% | 67.81% | 67.08% |
| Shoes                        | 72.40% | 76.90% | 74.58% |
| Trash                        | 80.70% | 50.55% | 62.16% |

The clothes category performed best with an F1-score of 88.27%.

The metal category was the most difficult category for the model, with an F1-score of 57.53%.

# 9. CONFUSION MATRIX

A confusion matrix was generated to examine how the model classified the different waste categories.

The matrix showed that some categories were classified correctly more often than others. Some misclassification occurred between visually similar categories such as:

* Metal and glass
* Paper and cardboard
* Plastic and glass
* Trash and other waste categories

This indicates that some waste materials have similar visual characteristics, making them more difficult for the CNN to distinguish.


# 10. PROJECT FILE STRUCTURE

The main project structure is:

AI_WASTE_CLASSIFICATION/
│
├── .gitignore
├── README.md
│
└── src/
    ├── dataset_distribution.py
    ├── inspect_dataset.py
    ├── view_sample.py
    ├── prepare_dataset.py
    ├── train.py
    └── evaluate_model.py

#Description

inspect_dataset.py
Used to inspect the dataset and count the images in each category.

dataset_distribution.py
Used to generate the dataset distribution graph.

view_sample.py
Used to view sample images from the dataset.

prepare_dataset.py
Used to prepare and split the dataset into training and validation sets.

train.py
Used to build and train the CNN model.

evaluate_model.py
Used to evaluate the trained model and generate performance metrics.

# 11. LIMITATIONS

The project has some limitations:

1. The dataset is not perfectly balanced.
2. Some categories contain significantly more images than others.
3. Some waste categories have similar visual characteristics.
4. The model achieved 73.30% accuracy, meaning there is still room for improvement.
5. The model was trained using the available computing resources.

# 12. FUTURE IMPROVEMENTS

The system could be improved by:

* Increasing the size of the dataset.
* Balancing the number of images in each category.
* Using more advanced CNN architectures.
* Applying transfer learning.
* Increasing data augmentation.
* Improving the classification of visually similar waste categories.
* Deploying the model as a web or mobile application.

# 13. CONCLUSION

This project successfully demonstrated the use of Artificial Intelligence and Machine Learning for waste image classification.

A Convolutional Neural Network was developed and trained using 12,259 images from ten waste categories. The dataset was divided into 9,802 training images and 2,457 validation images.

The trained model achieved a 73.30% validation accuracy, demonstrating that the CNN was able to learn useful visual features for classifying different types of waste.

Although the model performed well on several categories, some classes such as metal, paper, plastic, and trash were more challenging. Further improvements could be achieved through a larger and more balanced dataset, transfer learning, and more advanced CNN architectures.

Overall, the project demonstrates the potential of Artificial Intelligence as a tool for automated waste classification and improved waste management.

# 14. REFERENCES

1. TensorFlow Documentation. TensorFlow and Keras documentation.
2. Scikit-learn Documentation. Machine Learning evaluation metrics.
3. Kaggle. Waste Image Classification Dataset.
4. Python Documentation. Python programming language documentation.


# 15. GROUP GITHUB REPOSITORIES

### Frederick Gyabeng

https://github.com/freakynet/AI_WASTE_CLASSIFICATION

### Nkrumah Philimon

https://github.com/P798/AI_WAIST_-CLASSIFICATION_ASSIGNMENT

### Yahaya Abdul-Kudus

https://github.com/Abdul-Kudus816/ARTIFICIAL-INTELLIGENCE-ASSIGNMENT
