import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


# --------------------------------------------------
# App title
# --------------------------------------------------

st.title("Dry Bean Classification")

st.write(
    "Enter the bean measurements below to predict the bean class."
)


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

area = st.number_input("Area", min_value=0.0)

perimeter = st.number_input("Perimeter", min_value=0.0)

major_axis_length = st.number_input(
    "Major Axis Length",
    min_value=0.0
)

minor_axis_length = st.number_input(
    "Minor Axis Length",
    min_value=0.0
)

aspect_ratio = st.number_input(
    "Aspect Ratio",
    min_value=0.0
)

eccentricity = st.number_input(
    "Eccentricity",
    min_value=0.0
)

convex_area = st.number_input(
    "Convex Area",
    min_value=0.0
)

equiv_diameter = st.number_input(
    "Equivalent Diameter",
    min_value=0.0
)

extent = st.number_input(
    "Extent",
    min_value=0.0
)

solidity = st.number_input(
    "Solidity",
    min_value=0.0
)

roundness = st.number_input(
    "Roundness",
    min_value=0.0
)

compactness = st.number_input(
    "Compactness",
    min_value=0.0
)

shape_factor1 = st.number_input(
    "Shape Factor 1",
    min_value=0.0,
    format="%.6f"
)

shape_factor2 = st.number_input(
    "Shape Factor 2",
    min_value=0.0,
    format="%.6f"
)

shape_factor3 = st.number_input(
    "Shape Factor 3",
    min_value=0.0
)

shape_factor4 = st.number_input(
    "Shape Factor 4",
    min_value=0.0
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Bean Class"):

    # Create dataframe in exactly the same order
    # as the training dataset

    input_data = pd.DataFrame({
        "Area": [area],
        "Perimeter": [perimeter],
        "MajorAxisLength": [major_axis_length],
        "MinorAxisLength": [minor_axis_length],
        "AspectRation": [aspect_ratio],
        "Eccentricity": [eccentricity],
        "ConvexArea": [convex_area],
        "EquivDiameter": [equiv_diameter],
        "Extent": [extent],
        "Solidity": [solidity],
        "roundness": [roundness],
        "Compactness": [compactness],
        "ShapeFactor1": [shape_factor1],
        "ShapeFactor2": [shape_factor2],
        "ShapeFactor3": [shape_factor3],
        "ShapeFactor4": [shape_factor4]
    })


    # --------------------------------------------------
    # Apply same log transformation used during training
    # --------------------------------------------------

    skewed_features = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "ConvexArea",
        "EquivDiameter"
    ]

    for col in skewed_features:
        input_data[col] = np.log1p(input_data[col])


    # --------------------------------------------------
    # Apply same StandardScaler used during training
    # --------------------------------------------------

    input_scaled = scaler.transform(input_data)


    # --------------------------------------------------
    # Predict
    # --------------------------------------------------

    prediction = model.predict(input_scaled)


    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    st.success(f"Predicted Bean Class: {prediction[0]}")