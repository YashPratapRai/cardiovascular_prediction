import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Cardiovascular Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------

@st.cache_resource
def load_model():
    return joblib.load("best_cardio_xgboost_model.pkl")

model = load_model()

# ----------------------------
# Header
# ----------------------------

st.title("❤️ Cardiovascular Disease Prediction")
st.markdown(
    """
    Predict the risk of cardiovascular disease using an
    **XGBoost Machine Learning Model**.

    **Model Accuracy:** 73.17%  
    **ROC-AUC Score:** 0.80
    """
)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("📊 Model Information")

st.sidebar.info(
    """
    **Algorithm:** XGBoost

    **Accuracy:** 73.17%

    **ROC-AUC:** 0.80

    **Dataset:** 70,000 Patients
    """
)

# ----------------------------
# Input Form
# ----------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age (Years)",
        min_value=18,
        max_value=100,
        value=50
    )

    gender = st.selectbox(
        "Gender",
        [1, 2],
        format_func=lambda x:
        "Male" if x == 1 else "Female"
    )

    height = st.number_input(
        "Height (cm)",
        min_value=120,
        max_value=220,
        value=170
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=30,
        max_value=200,
        value=70
    )

    ap_hi = st.number_input(
        "Systolic BP",
        min_value=80,
        max_value=250,
        value=120
    )

    ap_lo = st.number_input(
        "Diastolic BP",
        min_value=40,
        max_value=200,
        value=80
    )

with col2:

    cholesterol = st.selectbox(
        "Cholesterol",
        [1, 2, 3],
        format_func=lambda x:
        {
            1: "Normal",
            2: "Above Normal",
            3: "Well Above Normal"
        }[x]
    )

    gluc = st.selectbox(
        "Glucose",
        [1, 2, 3],
        format_func=lambda x:
        {
            1: "Normal",
            2: "Above Normal",
            3: "Well Above Normal"
        }[x]
    )

    smoke = st.selectbox(
        "Smoking",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

    alco = st.selectbox(
        "Alcohol",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

    active = st.selectbox(
        "Physically Active",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

# ----------------------------
# Feature Engineering
# ----------------------------

bmi = weight / ((height / 100) ** 2)

pulse_pressure = ap_hi - ap_lo

map_value = (2 * ap_lo + ap_hi) / 3

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("BMI", f"{bmi:.2f}")
col2.metric("Pulse Pressure", f"{pulse_pressure:.0f}")
col3.metric("MAP", f"{map_value:.2f}")

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Risk"):

    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
        "map": map_value
    }])

    probability = model.predict_proba(input_df)[0][1]

    prediction = model.predict(input_df)[0]

    st.markdown("---")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Risk Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red"}
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    if prediction == 1:

        st.error(
            f"⚠️ High Cardiovascular Risk ({probability*100:.2f}%)"
        )

    else:

        st.success(
            f"✅ Low Cardiovascular Risk ({probability*100:.2f}%)"
        )

# ----------------------------
# Feature Importance
# ----------------------------

st.markdown("---")
st.subheader("Top Risk Factors Used By Model")

importance_df = pd.DataFrame({
    "Feature": [
        "ap_hi",
        "cholesterol",
        "map",
        "age",
        "active",
        "smoke",
        "alco",
        "bmi",
        "gluc",
        "weight"
    ],
    "Importance": [
        0.629,
        0.099,
        0.075,
        0.060,
        0.025,
        0.020,
        0.016,
        0.016,
        0.015,
        0.012
    ]
})

st.bar_chart(
    importance_df.set_index("Feature")
)

# ----------------------------
# Footer
# ----------------------------

st.markdown("---")

st.info(
    """
    This tool is for educational purposes only.
    It should not replace professional medical advice.
    """
)