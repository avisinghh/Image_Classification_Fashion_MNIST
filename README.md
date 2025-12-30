# Fashion-MNIST Image Classifier (Streamlit)

This repository contains a Streamlit-based web application for classifying clothing images using a Convolutional Neural Network (CNN) trained on the Fashion-MNIST dataset. The application allows users to upload images, visualize preprocessing steps, generate predictions with confidence scores, and evaluate the trained model on the official Fashion-MNIST test set.

---

## Overview

The goal of this project is to demonstrate an end-to-end deep learning workflow, including:

- Training a CNN on Fashion-MNIST
- Saving the trained model
- Deploying the model using a Streamlit web interface
- Performing real-time inference on uploaded images
- Evaluating performance using standard classification metrics

The application automatically preprocesses any input image to match the Fashion-MNIST format (28×28 grayscale) before prediction.

---

## Features

- Interactive Streamlit user interface
- Upload images in JPG, PNG, BMP, or WEBP format
- Automatic image preprocessing:
  - Resize to 28×28
  - Grayscale conversion
  - Optional inversion and contrast normalization
- Displays predicted class and confidence score
- Shows Top-K class probabilities
- Visualization of preprocessing output
- Evaluation on Fashion-MNIST test set:
  - Accuracy
  - Confusion matrix
  - Classification report
- Auto-reload of model when the saved model file is updated
- GPU support if available

---

## Fashion-MNIST Classes

The model predicts one of the following 10 classes:

- T-shirt
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

---

## Model Performance

Evaluation on the official Fashion-MNIST test set (10,000 images):

Accuracy: 90.00%

The model is a custom CNN implemented using TensorFlow and Keras.

---

## Project Structure
Fashion_Dataset/
├── fashion_predict_app.py # Streamlit application
├── fashion_cnn.keras # Trained CNN model
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── training_curves.png # Training and validation curves
├── confusion_matrix.png # Confusion matrix visualization
└── classification_report.txt # Detailed classification metrics
