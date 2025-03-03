import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# Load the trained model safely
model_path = "rainfall_prediction_model.h5"
if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' not found. Please check the file path.")
else:
    model = load_model(model_path)

# Load dataset for scaler
data_path = r"C:\Users\japer\OneDrive\Desktop\Rain_Fall_Pred_kaggle competition\Dataset\train.csv"
if not os.path.exists(data_path):
    st.error(f"❌ Dataset file '{data_path}' not found. Please check the file path.")
else:
    df_train = pd.read_csv(data_path)
    X_train = df_train.drop(columns=['id', 'day', 'rainfall'])

    # Initialize and fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Streamlit UI
    st.title("🌧️ Rainfall Prediction App")
    st.write("Enter the required features to predict the probability of rainfall.")

    # Input fields for features
    feature_values = []
    for col in X_train.columns:
        value = st.number_input(f"Enter value for {col}:", value=0.0)
        feature_values.append(value)

    # Predict button
    if st.button("Predict Rainfall Probability"):
        try:
            # Convert input to numpy array
            features = np.array(feature_values).reshape(1, -1)

            # Scale the input
            features_scaled = scaler.transform(features)

            # Reshape for LSTM or sequence-based model (if applicable)
            features_seq = features_scaled.reshape((1, 1, features_scaled.shape[1]))

            # Predict
            prediction = model.predict(features_seq)
            result = float(prediction[0][0])

            # Display result
            st.success(f"🌧️ Rainfall Probability: {result:.4f}")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {str(e)}")
