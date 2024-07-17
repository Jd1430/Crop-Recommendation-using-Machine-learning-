import streamlit as st
import pandas as pd
import pickle

# Load the trained model and scaler
with open("Crop Recommandation\crop_recommendation_model.pkl", 'rb') as f:
    model = pickle.load(f)

with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

st.title("Crop Recommendation System")
st.image("img1.jpg", width=710)
st.write("""
This is a simple Crop Recommendation System using a Random Forest Classifier.
""")

st.header("Input Parameters")

def user_input_features():
    N = st.number_input('Nitrogen content in soil', min_value=0, max_value=140, value=0)
    P = st.number_input('Phosphorus content in soil', min_value=0, max_value=140, value=0)
    K = st.number_input('Potassium content in soil', min_value=0, max_value=200, value=0)
    temperature = st.number_input('Temperature (in Celsius)', min_value=0.0, max_value=50.0, value=0.0)
    humidity = st.number_input('Humidity (in %)', min_value=0.0, max_value=100.0, value=0.0)
    ph = st.number_input('pH value of soil', min_value=0.0, max_value=14.0, value=0.0)
    rainfall = st.number_input('Rainfall (in mm)', min_value=0.0, max_value=300.0, value=0.0)
    data = {'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Parameters')
st.write(input_df)

# Scale the input features
scaled_input = scaler.transform(input_df)

# Predict crop recommendation
prediction = model.predict(scaled_input)

# Extract the first element from the prediction array
predicted_crop = prediction[0]

st.write(f"<div style='text-align: left; font-size: 50px;'>Recommended Crop: {predicted_crop}</div>", unsafe_allow_html=True)