import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
import requests
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from datetime import datetime

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


def generate_pdf_report(disease_name, desc, precautions, medications, diets, workouts):
    """
    Generate a PDF report for the predicted disease.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#2C3E50',
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor='#34495E',
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    disease_style = ParagraphStyle(
        'DiseaseStyle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor='#27AE60',
        spaceAfter=20,
        spaceBefore=10,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    body_style = styles['BodyText']
    body_style.fontSize = 11
    body_style.leading = 14
    
    # Add title
    title = Paragraph("📋 Full Disease Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add date
    date_text = f"<i>Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>"
    date_para = Paragraph(date_text, body_style)
    elements.append(date_para)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add info text
    info_text = "This section provides a complete disease report based on the latest prediction you made."
    info_para = Paragraph(info_text, body_style)
    elements.append(info_para)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add predicted disease
    disease_heading = Paragraph(f"🩺 Predicted Disease: {disease_name}", disease_style)
    elements.append(disease_heading)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add Description
    desc_heading = Paragraph("Description", heading_style)
    elements.append(desc_heading)
    desc_para = Paragraph(desc, body_style)
    elements.append(desc_para)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add Precautions
    prec_heading = Paragraph("Precautions", heading_style)
    elements.append(prec_heading)
    prec_items = [ListItem(Paragraph(p, body_style)) for p in precautions[0] if p]
    prec_list = ListFlowable(prec_items, bulletType='bullet')
    elements.append(prec_list)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add Medications
    med_heading = Paragraph("Medications", heading_style)
    elements.append(med_heading)
    med_items = [ListItem(Paragraph(str(m), body_style)) for m in medications if m]
    med_list = ListFlowable(med_items, bulletType='bullet')
    elements.append(med_list)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add Diet
    diet_heading = Paragraph("Diet", heading_style)
    elements.append(diet_heading)
    diet_items = [ListItem(Paragraph(str(d), body_style)) for d in diets if d]
    diet_list = ListFlowable(diet_items, bulletType='bullet')
    elements.append(diet_list)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add Workout
    workout_heading = Paragraph("Workout", heading_style)
    elements.append(workout_heading)
    workout_items = [ListItem(Paragraph(str(w), body_style)) for w in workouts if w]
    workout_list = ListFlowable(workout_items, bulletType='bullet')
    elements.append(workout_list)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add footer
    footer_text = "<i>Disclaimer: This report is generated by an AI system and should not replace professional medical advice.</i>"
    footer_para = Paragraph(footer_text, body_style)
    elements.append(Spacer(1, 0.3*inch))
    elements.append(footer_para)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

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
        
        # Generate PDF automatically
        pdf_buffer = generate_pdf_report(dis, desc, pre, med, die, wrkout)
        
        # Display report on screen
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
        
        # Automatic PDF Download Button (no extra click needed)
        st.markdown("---")
        st.download_button(
            label="📥 Download Report as PDF",
            data=pdf_buffer,
            file_name=f"{dis}_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )

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
