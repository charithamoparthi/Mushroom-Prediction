import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model("logistic_model.h5")

# Load the original dataset (used for encoding reference)
df = pd.read_csv("ass.csv")
features = df.drop("class", axis=1)
encoders = {col: LabelEncoder().fit(features[col]) for col in features.columns}

st.title("🍄 Mushroom Edibility Predictor")
st.write("Enter the properties of the mushroom to check if it's edible or poisonous.")

user_input = {}
for col in features.columns:
    options = encoders[col].classes_
    user_input[col] = st.selectbox(f"{col}", options)

if st.button("Predict"):
    input_encoded = [encoders[col].transform([user_input[col]])[0] for col in features.columns]
    input_array = np.array([input_encoded])
    prediction = model.predict(input_array)[0][0]

    result = "🍄 Poisonous" if prediction > 0.5 else "✅ Edible"
    st.subheader(f"Prediction: {result}")
