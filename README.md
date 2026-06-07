# Crop Disease Detection System

## Overview

This project is a deep learning-based application that detects crop diseases from leaf images. It uses a Convolutional Neural Network (CNN) trained on a plant disease dataset to classify different types of crop diseases.

The system allows users to upload an image or capture it using a camera, and it provides prediction results along with confidence scores and treatment suggestions.

---

## Features

* Multi-class disease classification (15 classes)
* Image upload and camera input
* Real-time prediction
* Confidence score display
* Top-3 predicted classes
* Disease description and treatment suggestions
* Interactive dashboard interface
* Downloadable prediction report

---

## Dataset

The model is trained on the PlantVillage dataset, which contains labeled images of plant leaves.

Dataset link:
https://www.kaggle.com/datasets/emmarex/plantdisease

---

## Installation

### Clone the repository

git clone <your-repository-link>
cd Crop_Disease_Detection_System

### Create virtual environment

python -m venv venv

### Activate environment

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### Install dependencies

pip install -r requirements.txt

---

## Dataset Preparation

Split the dataset into training, validation, and testing sets:
python split_dataset.py

---

## Model Training

Run the training script:
python train.py

This will generate trained model files and performance graphs.

---

## Run the Application

Start the Streamlit app:
streamlit run app.py

Open in browser:
http://localhost:8501

---

## Model Information

* Model type: Convolutional Neural Network (CNN)
* Input size: 128 x 128
* Output: Multi-class classification
* Optimizer: Adam
* Loss function: Categorical Crossentropy

---

## Handling Data Imbalance

* Data augmentation used for smaller classes
* Balanced dataset split
* Regularization applied to reduce overfitting

---

## Output

The system provides:

* Predicted disease name
* Confidence score
* Top-3 predictions
* Disease description
* Treatment recommendation
* Probability distribution chart

---

## Requirements

* Python 3.8 or higher
* TensorFlow
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Pillow
* Scikit-learn

---

## Future Scope

* Deployment on cloud platform
* Mobile application integration
* Real-time monitoring system
* Improved model accuracy

---

## Author

Gayatri Shirsath
