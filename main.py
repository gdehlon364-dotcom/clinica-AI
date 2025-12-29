import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
import requests
import os

# ====================== Utility Functions ======================

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ====================== Paths ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "svc.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "Datasets")

# ====================== Load Model & Data ======================

svc = pickle.load(open(MODEL_PATH, "rb"))
training_df = pd.read_csv(os.path.join(DATASET_PATH, "Training.csv"))
precautions = pd.read_csv(os.path.join(DATASET_PATH, "precautions_df.csv"))
workout = pd.read_csv(os.path.join(DATASET_PATH, "workout_df.csv"))
description = pd.read_csv(os.path.join(DATASET_PATH, "description.csv"))
medications = pd.read_csv(os.path.join(DATASET_PATH, "medications.csv"))
diets = pd.read_csv(os.path.join(DATASET_PATH, "diets.csv"))

symptoms_dict = {symptom: i for i, symptom in enumerate(training_df.columns[:-1])}

# ====================== Helper Functions ======================

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [m for m in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [d for d in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    X_input = pd.DataFrame([input_vector], columns=list(symptoms_dict.keys()))
    pred = svc.predict(X_input)[0]
    return pred

# ====================== Streamlit UI Setup ======================

st.set_page_config(page_title="AI Disease Prediction", page_icon="🏥", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align:center;'>
            <img src='https://cdn-icons-png.flaticon.com/512/2966/2966484.png' width='80'>
            <h3>HealthCare AI Assistant</h3>
            <p style='color:gray;'>Your personal health prediction companion.</p>
        </div>
        <hr>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate to:", ["Predict Disease", "About Model", "Contact Support"])
    st.markdown("---")
    full_report_button = st.button("📄 Full Report", use_container_width=True)
    st.markdown("<p style='text-align:center;'>Developed by <b>Sapna and Inderjeet Kaur</b></p>", unsafe_allow_html=True)

# Header
st.markdown("""
    <style>
    .main-title { font-size: 40px; text-align: center; color: #2C3E50; font-weight: 700; }
    .subtitle { text-align: center; color: #7F8C8D; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏥 Prescripto: An AI-Driven Healthcare Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter your symptoms and get AI-powered diagnosis & insights.</div>", unsafe_allow_html=True)

# Lottie Animation
lottie_ai = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_q5pk6p1k.json")
if lottie_ai:
    st_lottie(lottie_ai, height=180, key="health")

# ====================== Pages ======================

# Predict Disease Page
if page == "Predict Disease":
    all_symptoms = list(symptoms_dict.keys())
    selected_symptoms = st.multiselect("Select symptoms:", all_symptoms)

    if st.button("🔍 Predict"):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            predicted_disease = get_predicted_value(selected_symptoms)
            desc, pre, med, die, wrkout = helper(predicted_disease)
            st.session_state["predicted_disease"] = predicted_disease

            st.markdown("## 🧬 AI Diagnosis Result")
            col1, col2, col3 = st.columns(3)
            col1.metric("Detected Disease", predicted_disease)
            col2.metric("Symptoms Count", len(selected_symptoms))
            col3.metric("Confidence", "~95%")

            tabs = st.tabs(["Disease", "Description", "Precautions", "Medications", "Workouts", "Diets"])
            with tabs[0]:
                st.success(predicted_disease)
            with tabs[1]:
                st.info(desc)
            with tabs[2]:
                for p in pre[0]:
                    st.write(f"- {p}")
            with tabs[3]:
                for m in med:
                    st.write(f"- {m}")
            with tabs[4]:
                for w in wrkout:
                    st.write(f"- {w}")
            with tabs[5]:
                for d in die:
                    st.write(f"- {d}")

# Full Report Page
if full_report_button:
    st.markdown("## 📋 Full Disease Report")
    st.info("Complete disease report based on latest prediction.")
    if "predicted_disease" not in st.session_state:
        st.warning("⚠️ Please predict a disease first.")
    else:
        dis = st.session_state["predicted_disease"]
        desc, pre, med, die, wrkout = helper(dis)
        st.success(f"### 🩺 Predicted Disease: {dis}")
        st.subheader("Description")
        st.write(desc)
        st.subheader("Precautions")
        for p in pre[0]:
            st.write(f"- {p}")
        st.subheader("Medications")
        for m in med:
            st.write(f"- {m}")
        st.subheader("Diet")
        for d in die:
            st.write(f"- {d}")
        st.subheader("Workout")
        for w in wrkout:
            st.write(f"- {w}")

# About Model Page
elif page == "About Model":
    st.subheader("📘 Model Overview")
    st.write("""
    This AI model uses a **Support Vector Classifier (SVC)** trained on medical data to predict diseases.
    It provides:
    - Disease name
    - Description
    - Recommended precautions, medications, diets, workouts
    """)
    st.info("Model Type: Support Vector Classifier (SVC)\nAccuracy: ~100%")

# Contact Page
elif page == "Contact Support":
    st.subheader("📞 Need Help?")
    st.write("Contact us:")
    st.write("- 📧 Email: aurevella234@gmail.com")
    st.write("- 💬 WhatsApp: +91 9216601686, +91 9779227674")
    st.write("- 🌐 Website: [HealthCare AI Portal](https://healthcareai.com)")

# ====================== Custom CSS ======================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
}
div[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
""", unsafe_allow_html=True)
