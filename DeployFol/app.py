import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODELS AND PREPROCESSORS ----------------
logistic = joblib.load("logistic.pkl")
svm = joblib.load("svm.pkl")
dt = joblib.load("dt.pkl")
nb = joblib.load("nb.pkl")
scaler = joblib.load("scaler.pkl")
label = joblib.load("label_encoder.pkl")

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="Dry Bean Classification", layout="wide")
st.title("🌱 Dry Bean Classification System")
st.write("Enter bean features below and click **Predict** to see the result.")

# ---------------- INPUT SECTION ----------------
st.subheader("📥 Enter Bean Features")

# Two-column layout for input
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area", min_value=0.0, value=10000.0, step=1.0, format="%.6f")
    perimeter = st.number_input("Perimeter", min_value=0.0, value=400.0, step=1.0, format="%.6f")
    major = st.number_input("MajorAxisLength", min_value=0.0, value=200.0, step=0.01, format="%.6f")
    minor = st.number_input("MinorAxisLength", min_value=0.0, value=100.0, step=0.01, format="%.6f")
    aspect = st.number_input("AspectRation", min_value=0.0, value=2.0, step=0.001, format="%.6f")
    ecc = st.number_input("Eccentricity", min_value=0.0, value=0.7, step=0.001, format="%.6f")
    convex = st.number_input("ConvexArea", min_value=0.0, value=12000.0, step=1.0, format="%.6f")
    equiv = st.number_input("EquivDiameter", min_value=0.0, value=100.0, step=0.01, format="%.6f")

with col2:
    extent = st.number_input("Extent", min_value=0.0, value=0.75, step=0.001, format="%.6f")
    solidity = st.number_input("Solidity", min_value=0.0, value=0.95, step=0.001, format="%.6f")
    roundness = st.number_input("Roundness", min_value=0.0, value=0.7, step=0.001, format="%.6f")
    compact = st.number_input("Compactness", min_value=0.0, value=0.5, step=0.001, format="%.6f")
    sf1 = st.number_input("ShapeFactor1", min_value=0.0, value=0.007331506, step=0.000001, format="%.9f")
    sf2 = st.number_input("ShapeFactor2", min_value=0.0, value=0.003147289, step=0.000001, format="%.9f")
    sf3 = st.number_input("ShapeFactor3", min_value=0.0, value=0.834222388, step=0.000001, format="%.9f")
    sf4 = st.number_input("ShapeFactor4", min_value=0.0, value=0.998723889, step=0.000001, format="%.9f")

# ---------------- PREDICT BUTTON ----------------
if st.button("Predict"):

    # ---------------- CREATE DATAFRAME ----------------
    input_df = pd.DataFrame([{
        "Area": area,
        "Perimeter": perimeter,
        "MajorAxisLength": major,
        "MinorAxisLength": minor,
        "AspectRation": aspect,
        "Eccentricity": ecc,
        "ConvexArea": convex,
        "EquivDiameter": equiv,
        "Extent": extent,
        "Solidity": solidity,
        "roundness": roundness,
        "Compactness": compact,
        "ShapeFactor1": sf1,
        "ShapeFactor2": sf2,
        "ShapeFactor3": sf3,
        "ShapeFactor4": sf4
    }])

    # ---------------- SCALE INPUT ----------------
    input_scaled = scaler.transform(input_df)

    # ---------------- BEST MODEL PREDICTION ----------------
    best_model = svm  # automatically use best model
    prediction = best_model.predict(input_scaled)
    predicted_class = label.inverse_transform(prediction)[0]

    st.subheader("🔮 Prediction Result")
    st.success(f"Predicted Bean Class: **{predicted_class}**")

    # ---------------- SHOW MODEL ACCURACY ----------------
    st.subheader("📊 Model Accuracy Comparison")
    accuracy_data = pd.DataFrame({
        "Model": ["Logistic Regression", "SVM", "Decision Tree", "Naive Bayes"],
        "Accuracy": [0.92, 0.93, 0.80, 0.90]  # Replace with your actual test accuracies
    })
    st.table(accuracy_data)

    # ---------------- OPTIONAL: Show input data ----------------
    with st.expander("🔍 View Entered Input Data"):
        st.dataframe(input_df)
