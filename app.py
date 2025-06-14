import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model dan Tools
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Obesity Predictor", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Obesity Level Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan data di bawah ini untuk memprediksi tingkat obesitas seseorang.</p>", unsafe_allow_html=True)
st.markdown("---")

def encode_inputs(df):
    mapping = {
        'yes': 1, 'no': 0,
        'Male': 1, 'Female': 0,
        'Sometimes': 1, 'Frequently': 2, 'Always': 3,
        'Public_Transportation': 0, 'Walking': 1, 'Automobile': 2, 'Motorbike': 3, 'Bike': 4
    }
    df.replace(mapping, inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df

with st.form("obesity_form"):
    st.subheader("🔍 Input Data")
    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox("Gender", ['Male', 'Female'])
        Age = st.slider("Age", 10, 100, 25)
        Height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70)
        Weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=65.0)
        family_history_with_overweight = st.selectbox("Family History of Overweight", ['yes', 'no'])
        FAVC = st.selectbox("Frequent High Caloric Food Consumption", ['yes', 'no'])
        FCVC = st.slider("Frequency of Vegetable Consumption (1-3)", 1.0, 3.0, 2.0)
        NCP = st.slider("Number of Main Meals", 1.0, 4.0, 3.0)

    with col2:
        CAEC = st.selectbox("Consumption of Food Between Meals", ['no', 'Sometimes', 'Frequently', 'Always'])
        SMOKE = st.selectbox("Do you smoke?", ['yes', 'no'])
        CH2O = st.slider("Water Consumption (liters/day)", 1.0, 3.0, 2.0)
        SCC = st.selectbox("Calories Monitoring?", ['yes', 'no'])
        FAF = st.slider("Physical Activity Frequency", 0.0, 3.0, 1.0)
        TUE = st.slider("Time Using Technology Devices", 0.0, 3.0, 1.0)
        CALC = st.selectbox("Alcohol Consumption", ['no', 'Sometimes', 'Frequently', 'Always'])
        MTRANS = st.selectbox("Transportation Used", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

    submitted = st.form_submit_button("🚀 Submit Prediction")

    if submitted:
        data = pd.DataFrame([{
            'Gender': Gender,
            'Age': Age,
            'Height': Height,
            'Weight': Weight,
            'family_history_with_overweight': family_history_with_overweight,
            'FAVC': FAVC,
            'FCVC': FCVC,
            'NCP': NCP,
            'CAEC': CAEC,
            'SMOKE': SMOKE,
            'CH2O': CH2O,
            'SCC': SCC,
            'FAF': FAF,
            'TUE': TUE,
            'CALC': CALC,
            'MTRANS': MTRANS
        }])

        encoded_df = encode_inputs(data)
        scaled_df = scaler.transform(encoded_df)
        prediction = model.predict(scaled_df)
        label = label_encoder.inverse_transform(prediction)

        st.markdown("---")
        st.markdown(f"<div style='background-color:#f0f2f6; padding:20px; border-radius:10px;'>"
                    f"<h3 style='color:#333;'>📊 Hasil Prediksi:</h3>"
                    f"<h2 style='color:#4CAF50;'>➡️ {label[0]}</h2>"
                    f"</div>", unsafe_allow_html=True)
