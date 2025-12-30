
# Fashion-MNIST Image Classifier (Streamlit)

This repository contains a Streamlit-based web application for classifying clothing images using a Convolutional Neural Network (CNN) trained on the Fashion-MNIST dataset. The application supports image upload, preprocessing visualization, real-time prediction, and evaluation on the Fashion-MNIST test set.

---

## Overview

This project demonstrates a complete end-to-end deep learning workflow:

- Training a CNN on the Fashion-MNIST dataset
- Saving the trained model
- Deploying the model using a Streamlit web interface
- Performing inference on uploaded images
- Evaluating model performance using standard classification metrics

Any uploaded image is automatically preprocessed to match the Fashion-MNIST input format (28×28 grayscale).

---

## Features

- Streamlit-based interactive user interface
- Upload images in JPG, PNG, BMP, or WEBP format
- Automatic image preprocessing:
  - Resize to 28×28
  - Convert to grayscale
  - Optional color inversion
  - Optional contrast normalization
- Prediction with confidence score
- Top-K class probability display
- Visualization of the preprocessed image
- Model evaluation on Fashion-MNIST test set:
  - Accuracy
  - Confusion matrix
  - Classification report
- Auto-reload of model when the `.keras` file is updated
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

Accuracy: 89.36%

The model is a custom CNN implemented using TensorFlow and Keras.

---

## Project Structure

The repository is organized to clearly separate source code, model files, and generated outputs.

```

Fashion_Dataset/
├── fashion_predict_app.py        # Streamlit application
├── fashion_cnn.keras             # Trained CNN model
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── outputs/
├── training_curves.png       # Training and validation curves
├── confusion_matrix.png      # Confusion matrix visualization
└── classification_report.txt # Detailed classification metrics

````

The `outputs/` directory contains generated artifacts produced during model training and evaluation. These files are included for transparency, reproducibility, and result inspection, and are not part of the application logic itself.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Fashion_Dataset.git
cd Fashion_Dataset
````

### 2. Create and Activate Conda Environment

```bash
conda create -n ML_DL python=3.9 -y
conda activate ML_DL
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit app using:

```bash
streamlit run fashion_predict_app.py
```

Open the application in your browser at:

```
http://localhost:8501
```

---

## Usage

1. Upload an image or select a sample from the Fashion-MNIST test set.
2. Adjust preprocessing options such as inversion, contrast normalization, and resize mode.
3. Preview the preprocessed 28×28 grayscale image.
4. Generate predictions and view confidence scores.
5. Use the evaluation tab to inspect accuracy, confusion matrix, and classification report.

---

## Notes

* The model expects 28×28 grayscale images; all preprocessing is handled automatically.
* Inverting colors and applying contrast normalization improves predictions for real-world photos.
* Distinguishing between Shirt and T-shirt is the most challenging task due to visual similarity.
* The app includes safeguards for very large uploaded images.

---

## Future Improvements

* Add Grad-CAM or saliency map visualization
* Support batch image prediction
* Deploy on Streamlit Cloud or Hugging Face Spaces
* Compare CNN performance with Vision Transformer models
* Improve robustness to real-world images using domain adaptation

---

## Author

Avinash Singh
PhD, Computer Engineering
University of Alabama at Birmingham

