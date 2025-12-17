import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved objects
logi = joblib.load("logistic.pkl")
svm = joblib.load("svm.pkl")
dt = joblib.load("dt.pkl")
nb = joblib.load("nb.pkl")
scaler = joblib.load("scaler.pkl")
label = joblib.load("label_encoder.pkl")

# Accuracy values (use your actual values)
model_accuracy = {
    "Logistic Regression": 0.92,
    "SVM": 0.93,
    "Decision Tree": 0.80,
    "Naive Bayes": 0.90
}

models = {
    "Logistic Regression": logi,
    "SVM": svm,
    "Decision Tree": dt,
    "Naive Bayes": nb
}

best_model_name = max(model_accuracy, key=model_accuracy.get)
best_model = models[best_model_name]

# App UI
st.set_page_config(page_title="Dry Bean Classifier", layout="centered")
st.title("🌱 Dry Bean Classification System")
st.write("Predict the class of dry beans using machine learning.")

# Sidebar input
st.sidebar.header("Enter Bean Features")

def user_input():
    return np.array([
        st.sidebar.number_input("Area", 0.0),
        st.sidebar.number_input("Perimeter", 0.0),
        st.sidebar.number_input("MajorAxisLength", 0.0),
        st.sidebar.number_input("MinorAxisLength", 0.0),
        st.sidebar.number_input("AspectRation", 0.0),
        st.sidebar.number_input("Eccentricity", 0.0),
        st.sidebar.number_input("ConvexArea", 0.0),
        st.sidebar.number_input("EquivDiameter", 0.0),
        st.sidebar.number_input("Extent", 0.0),
        st.sidebar.number_input("Solidity", 0.0),
        st.sidebar.number_input("roundness", 0.0),
        st.sidebar.number_input("Compactness", 0.0),
        st.sidebar.number_input("ShapeFactor1", 0.0),
        st.sidebar.number_input("ShapeFactor2", 0.0),
        st.sidebar.number_input("ShapeFactor3", 0.0),
        st.sidebar.number_input("ShapeFactor4", 0.0)
    ]).reshape(1, -1)

input_data = user_input()

# Prediction
scaled_data = scaler.transform(input_data)
prediction = best_model.predict(scaled_data)
predicted_class = label.inverse_transform(prediction)[0]

st.subheader("🔮 Prediction Result")
st.success(f"Predicted Bean Class: **{predicted_class}**")

# Accuracy table
st.subheader("📊 Model Accuracy Comparison")
acc_df = pd.DataFrame.from_dict(model_accuracy, orient="index", columns=["Accuracy"])
st.table(acc_df)

st.info(f"🏆 Best Model Selected: **{best_model_name}**")
