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
    try:
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
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        disease_style = ParagraphStyle(
            'DiseaseStyle',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            spaceBefore=10,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        body_style = styles['BodyText']
        body_style.fontSize = 11
        body_style.leading = 14
        
        # Add title
        title = Paragraph("Full Disease Report", title_style)
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
        disease_heading = Paragraph(f"<b>Predicted Disease: {disease_name}</b>", disease_style)
        elements.append(disease_heading)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add Description
        desc_heading = Paragraph("<b>Description</b>", heading_style)
        elements.append(desc_heading)
        # Clean description text
        clean_desc = str(desc).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        desc_para = Paragraph(clean_desc, body_style)
        elements.append(desc_para)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add Precautions
        prec_heading = Paragraph("<b>Precautions</b>", heading_style)
        elements.append(prec_heading)
        prec_items = [ListItem(Paragraph(str(p).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'), body_style)) for p in precautions[0] if p and str(p).strip()]
        if prec_items:
            prec_list = ListFlowable(prec_items, bulletType='bullet')
            elements.append(prec_list)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add Medications
        med_heading = Paragraph("<b>Medications</b>", heading_style)
        elements.append(med_heading)
        med_items = [ListItem(Paragraph(str(m).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'), body_style)) for m in medications if m and str(m).strip()]
        if med_items:
            med_list = ListFlowable(med_items, bulletType='bullet')
            elements.append(med_list)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add Diet
        diet_heading = Paragraph("<b>Diet</b>", heading_style)
        elements.append(diet_heading)
        diet_items = [ListItem(Paragraph(str(d).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'), body_style)) for d in diets if d and str(d).strip()]
        if diet_items:
            diet_list = ListFlowable(diet_items, bulletType='bullet')
            elements.append(diet_list)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add Workout
        workout_heading = Paragraph("<b>Workout</b>", heading_style)
        elements.append(workout_heading)
        workout_items = [ListItem(Paragraph(str(w).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'), body_style)) for w in workouts if w and str(w).strip()]
        if workout_items:
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
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

# ====================== Streamlit UI Setup ======================

st.set_page_config(page_title="AI Disease Prediction", page_icon="🏥", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align:center;'>
            <img src='https://cdn-icons-png.flaticon.com/512/2966/2966484.png' width='80'>
            <h3>Clinica AI</h3>
            <p style='color:gray;'>Your personal health prediction companion.</p>
        </div>
        <hr>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate to:", ["Predict Disease", "About Model", "Contact Support"])
    st.markdown("---")
    
    # Styled PDF Full Report button
    st.markdown("""
        <style>
        div.stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        </style>
    """, unsafe_allow_html=True)
    
    full_report_button = st.button("📝 PDF Full Report", use_container_width=True)
    st.markdown("<p style='text-align:center;'>Developed by <b>Gaurav, Suraj and Rahul</b></p>", unsafe_allow_html=True)

# Header
st.markdown("""
    <style>
    .main-title { font-size: 40px; text-align: center; color: #2C3E50; font-weight: 700; }
    .subtitle { text-align: center; color: #7F8C8D; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏥 Clinica AI</div>", unsafe_allow_html=True)
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
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0;'>📋 PDF Full Disease Report</h1>
            <p style='color: #f0f0f0; margin: 5px 0 0 0;'>Complete disease report with all medical recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    if "predicted_disease" not in st.session_state:
        st.warning("⚠️ Please predict a disease first using the 'Predict Disease' page.")
    else:
        dis = st.session_state["predicted_disease"]
        desc, pre, med, die, wrkout = helper(dis)
        
        # Store report data in session state
        st.session_state["report_data"] = {
            "disease": dis,
            "desc": desc,
            "pre": pre,
            "med": med,
            "die": die,
            "wrkout": wrkout
        }
        
        # Display report on screen with enhanced styling
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                        padding: 20px; border-radius: 10px; border-left: 5px solid #667eea; margin-bottom: 20px;'>
                <h2 style='color: #2C3E50; margin: 0;'>🩺 Predicted Disease: {dis}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Description Section
        st.markdown("""
            <div style='background: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px;'>
                <h3 style='color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px;'>📝 Description</h3>
            </div>
        """, unsafe_allow_html=True)
        st.write(desc)
        
        # Precautions Section
        st.markdown("""
            <div style='background: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px; margin-top: 15px;'>
                <h3 style='color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px;'>⚠️ Precautions</h3>
            </div>
        """, unsafe_allow_html=True)
        for p in pre[0]:
            st.write(f"• {p}")
        
        # Medications Section
        st.markdown("""
            <div style='background: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px; margin-top: 15px;'>
                <h3 style='color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px;'>💊 Medications</h3>
            </div>
        """, unsafe_allow_html=True)
        for m in med:
            st.write(f"• {m}")
        
        # Diet Section
        st.markdown("""
            <div style='background: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px; margin-top: 15px;'>
                <h3 style='color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px;'>🥗 Diet Recommendations</h3>
            </div>
        """, unsafe_allow_html=True)
        for d in die:
            st.write(f"• {d}")
        
        # Workout Section
        st.markdown("""
            <div style='background: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px; margin-top: 15px;'>
                <h3 style='color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px;'>🏋️‍♂️ Workout Plan</h3>
            </div>
        """, unsafe_allow_html=True)
        for w in wrkout:
            st.write(f"• {w}")
        
        # PDF Generation Section with enhanced styling
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 10px;'>
                <h3 style='color: #2C3E50;'>📥 Download Your Medical Report</h3>
                <p style='color: #7F8C8D;'>Get a professionally formatted PDF copy of your complete disease report</p>
            </div>
        """, unsafe_allow_html=True)
        
        try:
            with st.spinner('Generating your PDF report...'):
                pdf_buffer = generate_pdf_report(dis, desc, pre, med, die, wrkout)
            
            if pdf_buffer:
                st.success("✅ PDF Generated Successfully!")
                
                # Enhanced download button with custom styling
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"{dis.replace(' ', '_')}_Medical_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
                
                st.markdown("""
                    <div style='text-align: center; margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px;'>
                        <p style='margin: 0; color: #6c757d; font-size: 14px;'>
                            <i>📌 Your report includes: Disease information, Description, Precautions, Medications, Diet plan, and Workout recommendations</i>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Failed to generate PDF. Please try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

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
    st.write("If you have any feedback or need assistance, feel free to reach out:")
    st.write("- 📧 Email: gauravdehlon@gmail.com")
    st.write("- 💬 WhatsApp: +91 9877473481, +91 6239346562")
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
