# Spam Email Detection

This project demonstrates how to build a machine learning model to classify emails as spam or ham (non-spam) using Natural Language Processing and Naive Bayes classification.

## Overview

This application uses a Gaussian Naive Bayes classifier trained on email data to detect spam messages. The model uses text preprocessing techniques including stopword removal, punctuation removal, and TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to extract features from email text. Users can input email content through an interactive web interface built with Streamlit to get real-time spam/ham predictions.

The project consists of two main components:

**Model Training (`spam_classifier_model.py`):**
- Load and preprocess the spam email dataset.
- Clean text data (lowercase, remove digits, punctuation, and stopwords).
- Extract features using CountVectorizer and TF-IDF transformation.
- Split data into training (70%) and testing (30%) sets.
- Train a Gaussian Naive Bayes classifier.
- Evaluate model performance with accuracy score and classification report.
- Save the trained model and vectorizers for deployment.

**Web Application (`app.py`):**
- Load the pre-trained model and vectorizers.
- Accept user input through a text area.
- Preprocess the input email text.
- Transform text using the trained vectorizer and TF-IDF.
- Predict and display whether the email is spam or ham.

## Requirements

- Python 3
- Required Python packages:
  - Streamlit
  - Pandas
  - NumPy
  - NLTK
  - scikit-learn

## Files

- `spam_classifier_model.py`: Script to train the spam detection model.
- `app.py`: Streamlit web application for spam email classification.
- `requirements.txt`: List of required Python packages.
- `spam.csv`: Dataset containing email messages and their labels (spam/ham).
- `spam_classifier.pkl`: Pre-trained model and vectorizers (generated after training).

## How to Run

### Step 1: Train the Model

1. Make sure you have the required packages installed:
   
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure `spam.csv` is in the same directory.
3. Run the training script:
   ```bash
   python spam_classifier_model.py
   ```
4. This will train the model and save `spam_classifier.pkl` containing the vectorizer, TF-IDF transformer, and trained model.

### Step 2: Run the Web Application

1. Ensure `spam_classifier.pkl` is generated from the training step.
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and navigate to the local URL (typically `http://localhost:8501`).
4. Enter an email message in the text area and click "Classify" to see if it's spam or ham.

## Results

- Interactive web interface for real-time spam detection
- Model accuracy and classification metrics printed during training
- Clear classification results (Spam/Ham) for user-provided email content

## Notes

- The model uses Gaussian Naive Bayes, which assumes feature independence.
- Text preprocessing includes lowercase conversion, digit removal, punctuation removal, and stopword filtering.
- For best results, ensure the training dataset (`spam.csv`) has columns named `Message` (email content) and `Category` (spam/ham labels).
- The model performance depends on the quality and size of the training dataset.

