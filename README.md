# AI Waste Classification Using Convolutional Neural Network

## 1. Project Overview

Waste management and proper waste sorting are important for effective recycling and environmental protection. This project applies **Artificial Intelligence and Machine Learning** to automatically classify waste materials from images.

A **Convolutional Neural Network (CNN)** was developed using **Python, TensorFlow, and Keras** to classify waste images into **10 different categories**.

The project demonstrates how Computer Vision and Deep Learning can be used to assist in automated waste classification.

---

## 2. Project Objectives

The main objectives of this project are to:

* Obtain and inspect a waste image dataset.
* Prepare the dataset for Machine Learning.
* Divide the dataset into training and validation sets.
* Develop a Convolutional Neural Network for image classification.
* Train the CNN using waste images.
* Evaluate the trained model.
* Analyze the model using accuracy, precision, recall, F1-score, and a confusion matrix.
* Save the trained model for future use.

---

## 3. Dataset

The project uses a waste image dataset obtained from **Kaggle**.

The dataset contains:

* **12,259 total images**
* **10 waste categories**
* **9,802 training images**
* **2,457 validation images**

### Waste Categories

The ten classes in the dataset are:

1. Battery
2. Biological
3. Cardboard
4. Clothes
5. Glass
6. Metal
7. Paper
8. Plastic
9. Shoes
10. Trash

---

## 4. Dataset Distribution

| Waste Category | Number of Images |
| -------------- | ---------------: |
| Battery        |              756 |
| Biological     |              699 |
| Cardboard      |            1,411 |
| Clothes        |            1,892 |
| Glass          |            1,736 |
| Metal          |              930 |
| Paper          |            1,336 |
| Plastic        |            1,597 |
| Shoes          |            1,449 |
| Trash          |              453 |
| **Total**      |       **12,259** |

The dataset is not perfectly balanced because the number of images varies between the different categories.

---

## 5. Machine Learning Method

### Convolutional Neural Network (CNN)

A **Convolutional Neural Network** was selected because the project involves image classification.

CNNs are capable of automatically learning visual features such as:

* Edges
* Shapes
* Textures
* Patterns
* Object features

These features are then used to classify an image into one of the predefined waste categories.

---

## 6. Technologies Used

The following technologies and Python libraries were used:

* **Python 3.13**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Git**
* **GitHub**
* **Visual Studio Code**

---

## 7. Dataset Preparation

The dataset was first inspected to verify the number of classes and images.

The images were then organized and divided into:

| Dataset    |     Images | Percentage |
| ---------- | ---------: | ---------: |
| Training   |      9,802 |        80% |
| Validation |      2,457 |        20% |
| **Total**  | **12,259** |   **100%** |

The images were standardized to a consistent input size before being used by the CNN.

---

## 8. Model Configuration

The main model configuration used in the project was:

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Algorithm         | Convolutional Neural Network    |
| Framework         | TensorFlow / Keras              |
| Number of Classes | 10                              |
| Image Size        | 128 × 128                       |
| Batch Size        | 32                              |
| Optimizer         | Adam                            |
| Loss Function     | Sparse Categorical Crossentropy |
| Maximum Epochs    | 30                              |
| Early Stopping    | Used                            |

---

## 9. Project Workflow

```text
                    Waste Dataset
                         |
                         ↓
                Dataset Inspection
                         |
                         ↓
                 Data Preparation
                         |
                         ↓
             Training / Validation Split
                         |
                         ↓
                 CNN Model Creation
                         |
                         ↓
                    Model Training
                         |
                         ↓
                  Model Evaluation
                         |
                         ↓
                Waste Classification
```

---

## 10. Model Training

The CNN was trained using the prepared training dataset.

During training, the model learned visual patterns associated with the ten waste categories.

The trained model was saved in Keras format as:

```text
waste_classifier.keras
```

The model was evaluated using the validation dataset after training.

---

## 11. Model Evaluation

The trained model achieved the following results:

| Metric                  |     Result |
| ----------------------- | ---------: |
| **Validation Accuracy** | **73.30%** |
| Validation Loss         | **0.8970** |
| Macro F1-Score          | **71.68%** |
| Weighted F1-Score       | **73.27%** |

The validation accuracy of **73.30%** means that the model correctly classified approximately 73 out of every 100 validation images.

---

## 12. Classification Report

| Class      | Precision | Recall |   F1-Score |
| ---------- | --------: | -----: | ---------: |
| Battery    |    84.33% | 74.34% |     79.02% |
| Biological |    77.60% | 69.29% |     73.21% |
| Cardboard  |    78.75% | 75.97% |     77.34% |
| Clothes    |    88.16% | 88.39% | **88.27%** |
| Glass      |    73.04% | 80.17% |     76.44% |
| Metal      |    57.53% | 57.53% | **57.53%** |
| Paper      |    59.30% | 63.06% |     61.12% |
| Plastic    |    66.36% | 67.81% |     67.08% |
| Shoes      |    72.40% | 76.90% |     74.58% |
| Trash      |    80.70% | 50.55% |     62.16% |

### Best Performing Class

The **clothes** category achieved the highest F1-score:

**88.27%**

### Most Challenging Class

The **metal** category achieved the lowest F1-score:

**57.53%**

This indicates that the model had more difficulty distinguishing metal images from some of the other waste categories.

---

## 13. Evaluation Techniques

The model was evaluated using:

* Accuracy
* Validation loss
* Precision
* Recall
* F1-score
* Confusion matrix

The confusion matrix was used to identify categories that were commonly confused by the model.

---

## 14. Project Structure

```text
AI_WASTE_CLASSIFICATION/
│
├── .gitignore
├── README.md
│
└── src/
    │
    ├── dataset_distribution.py
    ├── evaluate_model.py
    ├── inspect_dataset.py
    ├── prepare_dataset.py
    ├── train.py
    └── view_sample.py
```

### Python Files

#### `inspect_dataset.py`

Inspects the dataset and displays the number of images available in each class.

#### `dataset_distribution.py`

Generates a graph showing the distribution of images across the waste categories.

#### `view_sample.py`

Displays sample images from the dataset for visual inspection.

#### `prepare_dataset.py`

Prepares the dataset and divides the images into training and validation sets.

#### `train.py`

Creates and trains the CNN model and saves the trained model.

#### `evaluate_model.py`

Loads the trained model and evaluates it using the validation dataset. It generates the classification report and confusion matrix.

---

## 15. Installation

### Clone the Repository

```bash
git clone https://github.com/freakynet/AI_WASTE_CLASSIFICATION.git
```

Enter the project directory:

```bash
cd AI_WASTE_CLASSIFICATION
```

### Install Required Libraries

```bash
pip install tensorflow scikit-learn matplotlib seaborn
```

---

## 16. Running the Project

### Inspect Dataset

```bash
python src/inspect_dataset.py
```

### View Sample Images

```bash
python src/view_sample.py
```

### Prepare Dataset

```bash
python src/prepare_dataset.py
```

### Train Model

```bash
python src/train.py
```

### Evaluate Model

```bash
python src/evaluate_model.py
```

---

## 17. Limitations

The current project has some limitations:

* The dataset is not perfectly balanced.
* Some waste categories have fewer images than others.
* Some categories have visually similar characteristics.
* Metal, paper, plastic, and trash were more difficult for the model to classify.
* The current validation accuracy is 73.30%.
* The model was trained using the available computing resources.

---

## 18. Future Improvements

The project can be improved by:

* Increasing the size of the dataset.
* Balancing the number of images in each class.
* Applying more data augmentation techniques.
* Using transfer learning.
* Testing more advanced CNN architectures.
* Improving classification of visually similar categories.
* Deploying the model as a web application.
* Developing a mobile application for real-time waste classification.

---

## 19. Group Members

| No. | Name                   | Index Number | GitHub Repository                                                                                          |
| --- | ---------------------- | ------------ | ---------------------------------------------------------------------------------------------------------- |
| 1   | **Frederick Gyabeng**  | UEB3505123   | [AI_WASTE_CLASSIFICATION](https://github.com/freakynet/AI_WASTE_CLASSIFICATION)                            |
| 2   | **Nkrumah Philimon**   | UEB3510823   | [AI_WAIST_-CLASSIFICATION_ASSIGNMENT](https://github.com/P798/AI_WAIST_-CLASSIFICATION_ASSIGNMENT)         |
| 3   | **Yahaya Abdul-Kudus** | UEB3515723   | [ARTIFICIAL-INTELLIGENCE-ASSIGNMENT](https://github.com/Abdul-Kudus816/ARTIFICIAL-INTELLIGENCE-ASSIGNMENT) |

---

## 20. Conclusion

This project demonstrates the application of Artificial Intelligence and Computer Vision to automated waste classification.

A Convolutional Neural Network was trained using **12,259 images belonging to ten waste categories**. The dataset was divided into **9,802 training images** and **2,457 validation images**.

The trained model achieved a **73.30% validation accuracy**.

The results demonstrate that CNNs can learn useful visual patterns for classifying different types of waste. However, the differences in performance between classes also indicate opportunities for improvement through better dataset balancing, data augmentation, and more advanced Machine Learning techniques.

---

## 21. Repository

### Frederick Gyabeng

GitHub:

https://github.com/freakynet/AI_WASTE_CLASSIFICATION

### Nkrumah Philimon

GitHub:

https://github.com/P798/AI_WAIST_-CLASSIFICATION_ASSIGNMENT

### Yahaya Abdul-Kudus

GitHub:

https://github.com/Abdul-Kudus816/ARTIFICIAL-INTELLIGENCE-ASSIGNMENT
