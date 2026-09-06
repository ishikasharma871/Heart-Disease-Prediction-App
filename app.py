import streamlit as st
import joblib
import pandas as pd

# ---------- Page config ----------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #fff5f5 0%, #ffffff 40%);
    }

    .hero {
        background: linear-gradient(135deg, #ff5c72 0%, #ff8a5c 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(255, 92, 114, 0.25);
    }
    .hero h1 {
        font-size: 2.3rem;
        margin-bottom: 0.3rem;
        color: white;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin: 0;
    }

    .card {
        background: white;
        padding: 1.5rem 1.7rem;
        border-radius: 16px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        margin-bottom: 1.3rem;
        border: 1px solid #ffe3e3;
    }

    .result-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
        color: white;
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        box-shadow: 0 8px 24px rgba(238, 82, 83, 0.35);
    }
    .result-low {
        background: linear-gradient(135deg, #38ef7d, #11998e);
        color: white;
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        box-shadow: 0 8px 24px rgba(17, 153, 142, 0.35);
    }

    section[data-testid="stSidebar"] {
        background: #fff0f0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #ff5c72, #ff8a5c);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        font-size: 1rem;
        transition: 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(255, 92, 114, 0.4);
    }

    [data-testid="stMetricValue"] {
        color: #ee5253;
        font-weight: 700;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Load saved model, scaler, columns ----------
model = joblib.load("SVM_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ---------- Hero header ----------
st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Risk Predictor</h1>
    <p>An AI-powered clinical decision support tool. Enter the patient's health
    parameters to instantly assess their risk of heart disease, based on a
    Support Vector Machine model trained on real clinical data.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar inputs ----------
st.sidebar.markdown("## 🩺 Patient Information")
st.sidebar.caption("Fill in all fields, then click Predict.")

age = st.sidebar.slider("Age", 1, 120, 40)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
chest_pain = st.sidebar.selectbox(
    "Chest Pain Type", ["ATA", "NAP", "ASY", "TA"],
    help="ATA = Atypical Angina, NAP = Non-Anginal Pain, ASY = Asymptomatic, TA = Typical Angina"
)
resting_bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 0, 250, 120)
cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", 0, 600, 200)
fasting_bs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"], horizontal=True)
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.sidebar.radio("Exercise-Induced Angina", ["N", "Y"], horizontal=True)
oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", -3.0, 7.0, 0.0, step=0.1)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

predict_btn = st.sidebar.button("🔍 Predict Risk", use_container_width=True)

# ---------- Summary card ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📋 Patient Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Age", age)
c2.metric("Resting BP", f"{resting_bp} mmHg")
c3.metric("Cholesterol", f"{cholesterol} mg/dl")
c4.metric("Max HR", max_hr)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Prediction ----------
if predict_btn:
    input_dict = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "M" else 0,
        "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(input_scaled)[0][1]
        except Exception:
            proba = None

    st.markdown("### 🎯 Prediction Result")
    if prediction == 1:
        st.markdown(
            '<div class="result-high">⚠️ High Risk of Heart Disease<br>'
            '<span style="font-size:0.95rem; font-weight:400;">'
            'Recommendation: Please consult a cardiologist for a thorough evaluation.</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-low">✅ Low Risk of Heart Disease<br>'
            '<span style="font-size:0.95rem; font-weight:400;">'
            'Recommendation: Continue maintaining a healthy lifestyle.</span></div>',
            unsafe_allow_html=True
        )

    if proba is not None:
        st.write("")
        st.caption(f"Model confidence (risk probability): **{proba*100:.1f}%**")
        st.progress(proba)

    with st.expander("🔬 View Model Input Data"):
        st.dataframe(input_df, use_container_width=True)
else:
    st.info("👈 Enter the patient details in the sidebar and click **Predict Risk** to see the result.")

st.write("")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.85rem;'>"
    "Powered by Machine Learning | Model: Support Vector Machine (SVM)</p>",
    unsafe_allow_html=True
)
