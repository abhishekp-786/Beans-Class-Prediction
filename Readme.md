🌱 Dry Bean Classification using Machine Learning
📌 Project Overview

This project implements an end-to-end machine learning classification system to predict the type of dry bean using morphological features. Multiple classification models are trained and evaluated, and the best-performing model is automatically selected for final prediction.
An interactive Streamlit web application is used for deployment and real-time predictions.

Dataset Source: UCI Machine Learning Repository – Dry Bean Dataset

🎯 Objectives

Perform data preprocessing and outlier analysis

Visualize feature distributions and relationships

Train and compare multiple classification models

Select the best model based on accuracy

Deploy the model using an interactive UI

📂 Dataset Information

Total Features: 16 numerical features

Target Variable: Class

Total Classes: 7

Bean Classes:

SEKER

HOROZ

DERMASON

BARBUNYA

CALI

BOMBAY

SIRA

🔧 Data Preprocessing

Checked for missing values (none found)

Detected outliers using Interquartile Range (IQR)

Scaled features using StandardScaler

Encoded class labels using LabelEncoder

📊 Exploratory Data Analysis (EDA)

Class distribution analysis

Boxplots for feature-wise class variation

Scatter plots to study feature relationships

Correlation heatmap for numerical features

Shape descriptor comparison across classes

🤖 Machine Learning Models

The following models were implemented and evaluated:

Logistic Regression

Support Vector Machine (SVM)

Decision Tree

Naive Bayes

Evaluation Metrics:

Accuracy Score

Confusion Matrix

Classification Report

➡️ SVM achieved the highest accuracy and is used as the final prediction model.

🚀 Deployment

Models, scaler, and label encoder saved using joblib

Built an interactive Streamlit web application

User inputs feature values and clicks Predict

Application:

Automatically scales input data

Uses the best model for prediction

Displays predicted bean class

Shows accuracy of all trained models

🛠️ Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Streamlit

Joblib

▶️ How to Run the Application

Clone the repository

git clone <repository-url>
cd Dry-Bean-Classification


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py

📈 Results

Successfully classified dry bean varieties with high accuracy

SVM performed best among all models

Real-time predictions enabled through web UI

📌 Conclusion

This project demonstrates a complete machine learning workflow from data preprocessing and visualization to model training, evaluation, and deployment using a user-friendly interface.

📬 Feedback

Suggestions and feedback are welcome!