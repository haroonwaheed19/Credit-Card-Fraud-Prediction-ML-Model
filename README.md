💳 Credit Card Fraud Detection Model

This repository contains an ML model for detecting fraudulent transactions in credit card data using Logistic Regression. The model is trained on an imbalanced dataset and applies various techniques to improve accuracy and handle class imbalance.

📌 Project Overview

Credit card fraud detection is a crucial application of machine learning in financial security. This project applies supervised learning techniques to classify transactions as either:

✅ Legitimate (0) – Normal transactions

🚨 Fraudulent (1) – Suspicious transactions

🔹 Dataset

    The dataset used is the Credit Card Fraud Dataset from Kaggle.
    It contains 284,807 transactions, out of which only 492 are fraudulent (~0.17% fraud cases).
    The dataset has 30 numerical features, including V1-V28 (PCA-transformed features), Time, and Amount.

🛠️ Tech Stack

    Python
    NumPy
    Pandas
    Scikit-Learn
    Matplotlib & Seaborn (for visualization)

📊 Model Training Process

    Load and preprocess data
        Read the dataset using pandas.
        Handle missing values (if any).
        Normalize the Amount column using StandardScaler.

    Handle Class Imbalance
        The dataset is highly imbalanced (0.17% fraud cases).
        Techniques used:
        ✅ Undersampling: Taking a random sample of non-fraudulent transactions equal to the number of fraudulent ones.
        ✅ Oversampling: Using SMOTE (Synthetic Minority Over-sampling Technique).

    Train-Test Split
        Data is split into training (80%) and testing (20%) sets using train_test_split().

    Feature Scaling
        Apply StandardScaler to normalize features.

    Model Selection & Training
        Train a Logistic Regression model.

    Model Evaluation
        Evaluate accuracy

📂 Project Structure

📂 Credit-Fraud-Detection

│── 📄 credit_data.csv         # Credit card transactions dataset

│── 📄 fraud_detection.ipynb   # Jupyter Notebook for training & analysis

│── 📄 README.md               # Project Documentation


📈 Model Performance

Metric	Score
Accuracy	92%

🚀 Future Enhancements

✅ Implement Deep Learning (LSTMs, Autoencoders) for fraud detection.

✅ Use Anomaly Detection instead of classification for better generalization.

✅ Deploy the model with FastAPI or Streamlit for real-time predictions.

📌 References

    Kaggle - Credit Card Fraud Detection Dataset
    Scikit-learn Documentation

