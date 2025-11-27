import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

DOCTOR_NAME = "Dr. Kshitij Bhatnagar"
DATASET_PATH = "doctor_kshitij_cases.csv"

# =========================================================
#  TEMPERATURE CONVERSION FUNCTIONS
# =========================================================
def fahrenheit_to_celsius(f_temp):
    """Convert Fahrenheit to Celsius"""
    return (f_temp - 32) * 5/9

def celsius_to_fahrenheit(c_temp):
    """Convert Celsius to Fahrenheit"""
    return (c_temp * 9/5) + 32

# =========================================================
#  STYLING AND UI CONFIG - IMPROVED VISIBILITY
# =========================================================
def apply_custom_styles():
    st.markdown("""
    <style>
    /* ===== GLOBAL DARK THEME ===== */
    .main, .block-container {
        background-color: #0f172a !important;  /* Deep dark navy */
        color: #e2e8f0 !important;              /* Light text */
        padding-top: 1rem !important;
    }

    /* Remove streamlit default padding & white lines  */
    div[data-testid="stVerticalBlock"] > div {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Fix sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }

    /* ===== CARD STYLING ===== */
    .professional-card {
        background: #1e293b !important;     /* Darker card */
        padding: 1.5rem;
        border-radius: 10px;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        margin-bottom: 1.2rem;
    }

    .emergency-card {
        background: #7f1d1d !important;   /* Dark red */
        border-left: 6px solid #dc2626 !important;
    }

    .status-card {
        background: #1e293b !important;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #334155 !important;
        margin-bottom: 1.2rem;
        text-align: center;
    }

    .status-waiting {
        border-color: #d97706 !important;
        background: #451a03 !important;
    }

    .status-approved {
        border-color: #059669 !important;
        background: #064e3b !important;
    }

    /* ===== HEADERS =====*/
    .section-header {
        color: #f8fafc !important;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #334155;  /* MAKE IT DARK */
        padding-bottom: 0.3rem;
    }

    /* ===== FORM INPUTS ===== */
    /* Fix all input field visibility */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox select,
    .stTextArea textarea,
    .stNumberInput > div > div > input {
        background: #1e293b !important;
        color: white !important;
        border: 1px solid #475569 !important;
        border-radius: 6px;
        padding: 0.5rem !important;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stSelectbox select:focus,
    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
        background: #1e293b !important;
        color: white !important;
    }

    /* Fix number input background specifically */
    input[type="number"] {
        background: #1e293b !important;
        color: white !important;
    }

    /* Fix labels and text visibility - IMPROVED CONTRAST */
    .stNumberInput label,
    .stTextInput label,
    .stSelectbox label,
    .stTextArea label,
    .stCheckbox label,
    .stRadio label {
        color: #f1f5f9 !important;  /* Brighter white for better visibility */
        font-weight: 600 !important; /* Bolder font */
        font-size: 1.02rem !important;
    }

    /* Specific fix for radio buttons */
    .stRadio > div {
        color: #f1f5f9 !important;
    }
    
    .stRadio > label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }

    /* Radio button options visibility */
    .stRadio [data-testid="stMarkdownContainer"] p {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }

    /* ===== BUTTONS ===== */
    .stButton button {
        background: linear-gradient(90deg, #2563eb, #1e40af);
        color: white !important;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        padding: 0.6rem 1rem;
        transition: 0.2s ease-in-out;
    }
    .stButton button:hover {
        transform: scale(1.03);
        background: linear-gradient(90deg, #1e40af, #2563eb);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* ===== SIDEBAR ===== */
    .css-1d391kg, .css-1lcbmhc, .css-1outwn7 {
        background-color: #1e293b !important;
    }

    /* ===== METRIC CARDS ===== */
    [data-testid="metric-container"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
        padding: 1rem;
    }

    /* ===== CONVERSATION BOXES ===== */
    .chat-box {
        background: #1e293b;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        color: #e2e8f0;
    }
    .chat-emergency {
        background: #2d1112;
        border-left: 4px solid #dc2626;
    }

    /* ===== CHECKBOX STYLING ===== */
    .stCheckbox label {
        color: #f1f5f9 !important;  /* Brighter white */
        font-weight: 500 !important;
    }

    /* ===== RADIO BUTTONS ===== */
    .stRadio div[data-testid="stMarkdownContainer"] p {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }

    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {
        background: #1e293b !important;
        color: #f1f5f9 !important;  /* Brighter white */
        border: 1px solid #334155 !important;
        font-weight: 600;
    }

    /* ===== SCROLL BAR ===== */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 10px;
    }

    /* ===== TEMPERATURE CONVERTER ===== */
    .temp-converter {
        background: #1e293b !important;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #475569 !important;
        margin: 0.5rem 0;
        color: #e2e8f0;
        font-weight: 500;
    }

    /* ===== RISK INDICATORS ===== */
    .risk-low { border-left: 4px solid #059669 !important; }
    .risk-moderate { border-left: 4px solid #d97706 !important; }
    .risk-high { border-left: 4px solid #dc2626 !important; }
    .risk-very-high { border-left: 4px solid #7f1d1d !important; }

    /* ===== SPECIFIC FIX FOR HEART RATE, BP INPUTS ===== */
    div[data-testid="stNumberInput"] > div > div > input {
        background: #1e293b !important;
        color: white !important;
        border: 1px solid #475569 !important;
    }

    /* Ensure all number inputs are visible */
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }

    /* ===== IMPROVED LABEL VISIBILITY ===== */
    /* Make all form labels more visible */
    div[data-testid="stForm"] label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 1.02rem !important;
    }

    /* Specific fix for form sections */
    .stForm label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }

    /* Temperature section labels */
    div[data-testid="stVerticalBlock"] > div > div > div > p {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
#  EMERGENCY DETECTION
# =========================================================
def check_emergency_condition(temp, symptoms, vitals):
    """Check if patient condition requires emergency attention"""
    emergency_conditions = []
    
    # High fever emergency (more reasonable thresholds)
    if temp >= 40.0:  # 104°F equivalent - true medical emergency
        emergency_conditions.append(f"Very high fever ({temp}°C / {celsius_to_fahrenheit(temp):.1f}°F)")
    elif temp >= 39.0:  # 102.2°F - concerning but not critical
        emergency_conditions.append(f"High fever ({temp}°C / {celsius_to_fahrenheit(temp):.1f}°F)")
    
    # ENT emergencies
    if symptoms.get('ear_pain') and temp >= 38.5:
        emergency_conditions.append("Ear pain with fever - Possible acute otitis media")
    
    if symptoms.get('throat_pain') and symptoms.get('cough') and temp >= 38.5:
        emergency_conditions.append("Severe throat infection symptoms")
    
    # Cardiovascular emergencies
    if vitals.get('systolic_bp', 120) > 180 or vitals.get('diastolic_bp', 80) > 120:
        emergency_conditions.append("Hypertensive crisis")
    
    if vitals.get('systolic_bp', 120) < 90 or vitals.get('diastolic_bp', 80) < 60:
        emergency_conditions.append("Hypotensive emergency")
    
    if vitals.get('heart_rate', 80) > 140:
        emergency_conditions.append("Severe tachycardia")
    elif vitals.get('heart_rate', 80) > 120:
        emergency_conditions.append("Tachycardia")
    
    if vitals.get('heart_rate', 80) < 40:
        emergency_conditions.append("Severe bradycardia")
    elif vitals.get('heart_rate', 80) < 50:
        emergency_conditions.append("Bradycardia")
    
    # Chest pain emergency
    if symptoms.get('chest_pain'):
        emergency_conditions.append("Chest pain - Requires immediate evaluation")
    
    # Breathing difficulties
    if symptoms.get('breathing_difficulty'):
        emergency_conditions.append("Breathing difficulty - Emergency evaluation needed")
    
    return emergency_conditions

def get_emergency_advice(conditions):
    """Get emergency medical advice"""
    advice = []
    
    if any("fever" in cond.lower() for cond in conditions):
        advice.extend([
            "🆘 **IMMEDIATE MEASURES:**",
            "• Take Paracetamol 650mg (if no allergies) for fever",
            "• Use cold compresses on forehead and armpits",
            "• Stay hydrated with electrolyte solutions",
            "• Remove excess clothing, keep room ventilated"
        ])
    
    if any("ear" in cond.lower() for cond in conditions):
        advice.extend([
            "🆘 **EAR PAIN EMERGENCY:**",
            "• Take Ibuprofen 400mg for pain and inflammation (if no stomach issues)",
            "• Avoid water entry in ears",
            "• Use warm compress on affected ear",
            "• Do NOT use eardrops without prescription"
        ])
    
    if any("throat" in cond.lower() for cond in conditions):
        advice.extend([
            "🆘 **THROAT EMERGENCY:**",
            "• Gargle with warm salt water every 2 hours",
            "• Use throat lozenges with anesthetic",
            "• Stay hydrated with warm liquids",
            "• Avoid spicy and hard foods"
        ])
    
    if any("chest" in cond.lower() for cond in conditions):
        advice.extend([
            "🆘 **CHEST PAIN PROTOCOL:**",
            "• SIT UPRIGHT immediately, do not lie down",
            "• If available, take Aspirin 325mg (if no allergies)",
            "• Loosen tight clothing",
            "• CALL EMERGENCY SERVICES if pain radiates to arm/jaw"
        ])
    
    if any("blood pressure" in cond.lower() for cond in conditions):
        advice.extend([
            "🆘 **BLOOD PRESSURE CRISIS:**",
            "• Sit quietly in a calm environment",
            "• Avoid any physical exertion",
            "• Do not take extra medication unless prescribed",
            "• Monitor BP every 15 minutes"
        ])
    
    if any("breathing" in cond.lower() for cond in conditions):
        advice.extend([
            "🆘 **BREATHING EMERGENCY:**",
            "• Sit upright in a comfortable position",
            "• Use pursed-lip breathing technique",
            "• Stay calm and try to relax",
            "• Use emergency inhaler if prescribed"
        ])
    
    advice.extend([
        "",
        "🚨 **URGENT: Proceed to nearest emergency department if:**",
        "• Symptoms worsen rapidly",
        "• Difficulty breathing occurs",
        "• Severe pain persists after medication",
        "• Altered mental state or confusion"
    ])
    
    return "\n".join(advice)

# =========================================================
#  RULE-BASED MEDICAL ADVICE SYSTEM (NO API NEEDED)
# =========================================================
def generate_medical_advice(patient_data, risk_score, risk_level, is_emergency=False, emergency_conditions=None):
    """
    Generate medical advice based on rules and symptoms without needing API
    """
    
    if is_emergency:
        emergency_advice = get_emergency_advice(emergency_conditions)
        
        advice = f"""
🚨 **EMERGENCY MEDICAL GUIDANCE** 🚨

**Critical Conditions Detected:**
{', '.join(emergency_conditions)}

{emergency_advice}

**Additional Emergency Notes:**
• Do not hesitate to call emergency services (911/112/108)
• Keep your phone charged and accessible
• Have someone stay with you if possible
• Bring all current medications to the hospital
• Inform emergency personnel of all symptoms

**⚠️ IMPORTANT: This is emergency guidance. Seek immediate medical attention.**
"""
    else:
        # Rule-based advice based on symptoms and risk
        symptoms_advice = []
        precautions = []
        medications = []
        
        # Temperature advice
        temp = patient_data.get('temperature', 37)
        if temp >= 39.0:
            symptoms_advice.append("• High fever detected - focus on fever management")
            medications.append("• Paracetamol 650mg every 6 hours for fever")
        elif temp >= 38.0:
            symptoms_advice.append("• Mild fever - monitor temperature regularly")
            medications.append("• Paracetamol 650mg as needed for fever above 38.5°C")
        
        # Cough and throat
        if patient_data.get('cough', False) and patient_data.get('throat_pain', False):
            symptoms_advice.append("• Cough with throat pain suggests upper respiratory infection")
            medications.append("• Throat lozenges and warm salt water gargles")
            medications.append("• Cough suppressant if cough is dry and bothersome")
        
        # Ear pain
        if patient_data.get('ear_pain', False):
            symptoms_advice.append("• Ear pain may indicate infection - avoid water entry")
            medications.append("• Ibuprofen 400mg for pain relief (if no stomach issues)")
        
        # Headache and body pain
        if patient_data.get('headache', False) or patient_data.get('body_pain', False):
            symptoms_advice.append("• General body aches and headache - ensure proper rest")
            medications.append("• Paracetamol 650mg or Ibuprofen 400mg for pain relief")
        
        # General precautions
        precautions.extend([
            "• Get plenty of rest and sleep",
            "• Stay hydrated with water, soups, and electrolyte solutions",
            "• Eat light, nutritious meals",
            "• Maintain good hygiene to prevent spread",
            "• Monitor symptoms daily"
        ])
        
        # Risk-based recommendations
        if risk_score >= 70:
            visit_timing = "**Visit Recommendation:** Consult doctor within 24 hours"
            urgency = "High priority - schedule appointment soon"
        elif risk_score >= 40:
            visit_timing = "**Visit Recommendation:** Consult doctor within 2-3 days if no improvement"
            urgency = "Moderate priority - monitor closely"
        else:
            visit_timing = "**Visit Recommendation:** Routine follow-up if symptoms persist beyond 5 days"
            urgency = "Low priority - self-care recommended"
        
        advice = f"""
📋 **MEDICAL GUIDANCE BASED ON SYMPTOMS**

**Risk Assessment:** {risk_score:.1f}/100 ({risk_level})

**Primary Symptoms Analysis:**
{chr(10).join(symptoms_advice) if symptoms_advice else "• General symptomatic care recommended"}

**Self-Care Measures:**
{chr(10).join(precautions)}

**Medication Suggestions (OTC):**
{chr(10).join(medications) if medications else "• No specific medications recommended at this time"}

{visit_timing}
{urgency}

**Follow-up Instructions:**
• Return if symptoms worsen or new symptoms develop
• Complete any prescribed course of medication
• Maintain follow-up as directed

**Note:** This is preliminary guidance based on symptom patterns. 
Final diagnosis and prescription should come from your healthcare provider.
"""

    return advice

# =========================================================
#  ADVANCED ML MODEL TRAINING (ENSEMBLE + HYPERPARAMETERS)
# =========================================================
@st.cache_resource(show_spinner=True)
def train_ml_model(dataset_path: str):
    """
    Advanced ML model using ensemble methods with feature engineering
    """
    try:
        df = pd.read_csv(dataset_path)
        
        # Feature Engineering
        df['bp_ratio'] = df['systolic_bp'] / df['diastolic_bp']
        df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['systolic_bp'] - df['diastolic_bp']) / 3
        df['symptom_count'] = df[['cough', 'throat_pain', 'ear_pain', 'chest_pain', 'headache', 'body_pain']].sum(axis=1)
        df['fever_indicator'] = (df['temperature'] > 38.0).astype(int)
        
        # Features & target
        feature_cols = [
            "age", "gender", "temperature",
            "systolic_bp", "diastolic_bp", "heart_rate",
            "cough", "throat_pain", "ear_pain",
            "chest_pain", "headache", "body_pain",
            "duration_days", "bp_ratio", "mean_arterial_pressure",
            "symptom_count", "fever_indicator"
        ]
        target_col = "risk_score"

        X = df[feature_cols]
        y = df[target_col]

        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

        # Train–test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5)
        )

        # Advanced Ensemble Model
        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        svm_model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel='rbf', C=2.0, epsilon=0.05, gamma='scale'))
        ])
        
        ridge_model = Ridge(alpha=1.0, random_state=42)

        # Voting Regressor (Ensemble)
        ensemble_model = VotingRegressor([
            ('rf', rf_model),
            ('gb', gb_model),
            ('svm', svm_model),
            ('ridge', ridge_model)
        ], weights=[3, 2, 1, 1])  # Higher weights for tree-based models

        # Train ensemble model
        ensemble_model.fit(X_train, y_train)
        
        # Get predictions from ensemble
        y_pred_ensemble = ensemble_model.predict(X_test)
        
        # Calculate ensemble metrics
        mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
        rmse_ensemble = mse_ensemble ** 0.5
        mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
        r2_ensemble = r2_score(y_test, y_pred_ensemble)

        best_metrics = {
            "rmse": rmse_ensemble,
            "mae": mae_ensemble,
            "r2": r2_ensemble,
            "model_name": "Advanced Ensemble (RF+GB+SVM+Ridge)",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "feature_importance": dict(zip(feature_cols, ensemble_model.estimators_[0].feature_importances_))
        }

        return ensemble_model, best_metrics, feature_cols

    except Exception as e:
        st.error(f"Error in model training: {e}")
        # Fallback to simple model
        return train_fallback_model(dataset_path)

def train_fallback_model(dataset_path: str):
    """Fallback model in case advanced training fails"""
    df = pd.read_csv(dataset_path)
    
    feature_cols = [
        "age", "gender", "temperature",
        "systolic_bp", "diastolic_bp", "heart_rate",
        "cough", "throat_pain", "ear_pain",
        "chest_pain", "headache", "body_pain",
        "duration_days"
    ]
    target_col = "risk_score"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    best_metrics = {
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "model_name": "Fallback Random Forest",
        "n_train": len(X_train),
        "n_test": len(X_test)
    }

    return model, best_metrics, feature_cols

# =========================================================
#  UTIL FUNCTIONS
# =========================================================
def map_risk_level(score: float) -> tuple:
    """Improved risk level mapping with better thresholds"""
    if score < 20:
        return "Low Risk", "risk-low"
    elif score < 40:
        return "Mild Risk", "risk-low"
    elif score < 60:
        return "Moderate Risk", "risk-moderate"
    elif score < 80:
        return "High Risk", "risk-high"
    else:
        return "Very High Risk", "risk-very-high"

def generate_summary_md(conversation, patient_info, prediction_info, is_emergency=False):
    """
    Creates a markdown consultation summary
    """
    patient_name = patient_info.get("name", "N/A")
    risk_score = prediction_info.get("risk_score", 0.0)
    risk_level = prediction_info.get("risk_level", "N/A")

    emergency_banner = "🚨 **EMERGENCY CONSULTATION** 🚨\n\n" if is_emergency else ""

    md = f"""
# 📝 Consultation Summary

{emergency_banner}
**Patient Name:** {patient_name}  
**Age:** {patient_info.get('age','-')}  
**Gender:** {patient_info.get('gender_display','-')}  

**Predicted Risk Score:** {risk_score:.1f} / 100  
**Risk Level:** {risk_level}  

---

## Conversation Log
"""
    for speaker, msg, is_emergency_msg in conversation:
        if is_emergency_msg:
            md += f"\n**🚨 {speaker}:**\n{msg}\n"
        else:
            md += f"\n**{speaker}:**\n{msg}\n"

    md += """
---

**Note:** This is an AI-assisted draft based on historical data.  
Final decision and prescription will always be given by the real doctor.
"""
    return md

# =========================================================
#  STREAMLIT APP
# =========================================================
def main():
    st.set_page_config(
        page_title=f"{DOCTOR_NAME} – AI Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_styles()

    # ========== TRY LOADING + TRAINING MODEL ==========
    try:
        model, metrics, feature_cols = train_ml_model(DATASET_PATH)
    except Exception as e:
        st.error(f"Error loading/training model from `{DATASET_PATH}`: {e}")
        st.stop()

    # SESSION STATE INIT
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "patient_info" not in st.session_state:
        st.session_state.patient_info = {}
    if "prediction_info" not in st.session_state:
        st.session_state.prediction_info = {}
    if "consultation_done" not in st.session_state:
        st.session_state.consultation_done = False
    if "waiting_for_approval" not in st.session_state:
        st.session_state.waiting_for_approval = False
    if "approval_status" not in st.session_state:
        st.session_state.approval_status = None
    if "is_emergency" not in st.session_state:
        st.session_state.is_emergency = False
    if "emergency_conditions" not in st.session_state:
        st.session_state.emergency_conditions = []

    # ----------- SIDEBAR -----------
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: #f8fafc; margin: 0;'>🩺 MEDICAL AI</h2>
            <p style='color: #94a3b8; margin: 0; font-size: 0.9rem;'>Advanced Ensemble ML · Rule-Based Advice</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ROLE SELECTION - IMPROVED VISIBILITY
        st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 1rem;'>Select Role:</h3>", unsafe_allow_html=True)
        role = st.radio(
            "",
            ["Patient", "Doctor"],
            label_visibility="collapsed",
            key="role_selector"
        )
        
        st.markdown("---")
        
        with st.expander("📊 AI Model Metrics", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{metrics['r2']:.3f}")
                st.metric("Model", metrics.get('model_name', 'Ensemble'))
            with col2:
                st.metric("RMSE", f"{metrics['rmse']:.2f}")
                st.metric("Training Samples", metrics['n_train'])
            
            # Show feature importance if available
            if 'feature_importance' in metrics:
                with st.expander("Top Features"):
                    top_features = sorted(metrics['feature_importance'].items(), 
                                        key=lambda x: x[1], reverse=True)[:5]
                    for feature, importance in top_features:
                        st.write(f"• {feature}: {importance:.3f}")
        
        with st.expander("ℹ️ Quick Guide", expanded=False):
            st.markdown("""
            **For Patients:**
            - Fill health details accurately
            - Emergency symptoms get priority
            - AI provides initial guidance only
            
            **For Doctors:**
            - Review AI recommendations
            - Final decision always with doctor
            """)

    # ----------- MAIN HEADER -----------
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='color: #f8fafc; margin: 0; font-size: 2.5rem;'>⚕ ADVANCED MEDICAL AI CONSULTATION</h1>
        <p style='color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
        Ensemble ML · Rule-Based Medical Advice · No API Required
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main layout
    col_main, col_side = st.columns([2.5, 1], gap="large")

    # =====================================================
    #                 PATIENT PORTAL
    # =====================================================
    if role == "Patient":
        with col_main:
            # Emergency Alert if conditions detected
            if st.session_state.emergency_conditions:
                st.markdown(f"""
                <div class='professional-card emergency-card'>
                    <h3 style='color: #f8fafc; margin: 0 0 0.5rem 0;'>🚨 Emergency Alert</h3>
                    <p style='margin: 0; color: #f8fafc;'><strong>Critical conditions detected:</strong></p>
                    <ul style='margin: 0.5rem 0 0 0; color: #f8fafc;'>
                        {''.join([f'<li>{condition}</li>' for condition in st.session_state.emergency_conditions])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Consultation Status Card
            if st.session_state.waiting_for_approval:
                st.markdown(f"""
                <div class='status-card status-waiting'>
                    <h3 style='color: #fbbf24; margin: 0 0 1rem 0;'>⏳ Consultation Sent for Review</h3>
                    <p style='color: #e2e8f0; margin: 0.5rem 0; font-size: 1.1rem;'>
                    <strong>Your consultation has been sent to {DOCTOR_NAME}</strong>
                    </p>
                    <p style='color: #94a3b8; margin: 0;'>
                    The doctor will review your case and provide final approval. <br>
                    You will be notified when your prescription is ready.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            elif st.session_state.approval_status is True:
                st.markdown(f"""
                <div class='status-card status-approved'>
                    <h3 style='color: #10b981; margin: 0 0 1rem 0;'>✅ Consultation Approved by Doctor</h3>
                    <p style='color: #e2e8f0; margin: 0.5rem 0; font-size: 1.1rem;'>
                    <strong>Your prescription has been reviewed and approved!</strong>
                    </p>
                    <p style='color: #94a3b8; margin: 0;'>
                    You can now safely follow the medical advice below. <br>
                    This consultation is supervised and finalized by {DOCTOR_NAME}.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            elif st.session_state.approval_status is False:
                st.markdown(f"""
                <div class='status-card'>
                    <h3 style='color: #ef4444; margin: 0 0 1rem 0;'>❌ Offline Consultation Required</h3>
                    <p style='color: #e2e8f0; margin: 0.5rem 0; font-size: 1.1rem;'>
                    <strong>Please visit the clinic for physical examination</strong>
                    </p>
                    <p style='color: #94a3b8; margin: 0;'>
                    The doctor has recommended an in-person consultation <br>
                    for better diagnosis and treatment.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Patient Form
            with st.form("patient_form", clear_on_submit=False):
                # Personal Information - IMPROVED VISIBILITY
                st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>👤 Personal Information</div>", unsafe_allow_html=True)
                
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    name = st.text_input("**Full Name**", 
                                       value=st.session_state.patient_info.get("name", ""),
                                       placeholder="Enter your full name")
                with col_p2:
                    age = st.number_input("**Age**", min_value=1, max_value=120, value=30)
                with col_p3:
                    gender_display = st.selectbox("**Gender**", ["Male", "Female", "Other"])
                st.markdown("</div>", unsafe_allow_html=True)

                # Vitals & Measurements - IMPROVED VISIBILITY
                st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>💓 Vitals & Measurements</div>", unsafe_allow_html=True)
                
                # Temperature with converter
                st.markdown("<strong style='color: #f1f5f9;'>Temperature</strong>", unsafe_allow_html=True)
                temp_col1, temp_col2 = st.columns([2, 1])
                with temp_col1:
                    temp_value = st.number_input("Value", min_value=30.0, max_value=110.0, value=98.6, step=0.1, label_visibility="collapsed")
                with temp_col2:
                    temp_unit = st.selectbox("Unit", ["°F", "°C"], label_visibility="collapsed")
                
                # Convert temperature
                if temp_unit == "°F":
                    temp_celsius = fahrenheit_to_celsius(temp_value)
                    temp_fahrenheit = temp_value
                    st.markdown(f"""
                    <div class='temp-converter'>
                        <small>{temp_value}°F = {temp_celsius:.1f}°C</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    temp_celsius = temp_value
                    temp_fahrenheit = celsius_to_fahrenheit(temp_value)
                    st.markdown(f"""
                    <div class='temp-converter'>
                        <small>{temp_value}°C = {temp_fahrenheit:.1f}°F</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Other vitals - IMPROVED VISIBILITY
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    heart_rate = st.number_input("**Heart Rate (bpm)**", 40, 200, 80)
                with col_v2:
                    sys_bp = st.number_input("**Systolic BP (mmHg)**", 80, 220, 120)
                with col_v3:
                    dias_bp = st.number_input("**Diastolic BP (mmHg)**", 40, 140, 80)
                
                duration = st.number_input("**Days since symptoms started**", 0, 30, 1)
                st.markdown("</div>", unsafe_allow_html=True)

                # Symptoms - IMPROVED VISIBILITY
                st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>🤒 Symptoms Checklist</div>", unsafe_allow_html=True)
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    cough = st.checkbox("Cough")
                    throat_pain = st.checkbox("Throat Pain")
                with col_s2:
                    ear_pain = st.checkbox("Ear Pain")
                    headache = st.checkbox("Headache")
                with col_s3:
                    chest_pain = st.checkbox("Chest Pain")
                    body_pain = st.checkbox("Body Pain")
                    breathing_difficulty = st.checkbox("Breathing Difficulty")
                st.markdown("</div>", unsafe_allow_html=True)

                # Additional Information
                st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📝 Additional Information</div>", unsafe_allow_html=True)
                free_text = st.text_area(
                    "Describe your symptoms in detail:",
                    placeholder="Example: I've had fever since yesterday evening, with ear pain and headache...",
                    height=100,
                    label_visibility="collapsed"
                )
                st.markdown("</div>", unsafe_allow_html=True)

                # Submit Button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    submitted = st.form_submit_button("**🔍 Get Medical Consultation**", 
                                                    use_container_width=True,
                                                    type="primary")

            if submitted:
                # Save patient info
                st.session_state.patient_info = {
                    "name": name,
                    "age": age,
                    "gender_display": gender_display,
                }

                gender = 1 if gender_display == "Male" else 0

                # Feature engineering for prediction
                bp_ratio = sys_bp / dias_bp if dias_bp > 0 else 1.5
                mean_arterial_pressure = dias_bp + (sys_bp - dias_bp) / 3
                symptom_count = sum([cough, throat_pain, ear_pain, chest_pain, headache, body_pain])
                fever_indicator = 1 if temp_celsius > 38.0 else 0

                # Build feature vector (using Celsius for model)
                feature_row = {
                    "age": age,
                    "gender": gender,
                    "temperature": temp_celsius,  # Use Celsius for model
                    "systolic_bp": sys_bp,
                    "diastolic_bp": dias_bp,
                    "heart_rate": heart_rate,
                    "cough": int(cough),
                    "throat_pain": int(throat_pain),
                    "ear_pain": int(ear_pain),
                    "chest_pain": int(chest_pain),
                    "headache": int(headache),
                    "body_pain": int(body_pain),
                    "duration_days": duration,
                    "bp_ratio": bp_ratio,
                    "mean_arterial_pressure": mean_arterial_pressure,
                    "symptom_count": symptom_count,
                    "fever_indicator": fever_indicator
                }

                # Check for emergency conditions (using Celsius)
                symptoms_dict = {
                    'ear_pain': ear_pain,
                    'throat_pain': throat_pain,
                    'cough': cough,
                    'chest_pain': chest_pain,
                    'breathing_difficulty': breathing_difficulty
                }
                
                vitals_dict = {
                    'systolic_bp': sys_bp,
                    'diastolic_bp': dias_bp,
                    'heart_rate': heart_rate
                }
                
                emergency_conditions = check_emergency_condition(temp_celsius, symptoms_dict, vitals_dict)
                st.session_state.emergency_conditions = emergency_conditions
                st.session_state.is_emergency = len(emergency_conditions) > 0

                # Prepare data for prediction
                X_new = pd.DataFrame([feature_row])
                
                # Ensure all required features are present
                for col in feature_cols:
                    if col not in X_new.columns:
                        X_new[col] = 0  # Default value for missing engineered features

                X_new = X_new[feature_cols]

                # ML prediction
                risk_score = float(model.predict(X_new)[0])
                risk_score = max(0.0, min(100.0, risk_score))
                risk_level, risk_class = map_risk_level(risk_score)

                st.session_state.prediction_info = {
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "features": feature_row,
                }

                # Generate medical advice using rule-based system (NO API)
                with st.spinner("🔬 Analyzing with Medical AI..." if not st.session_state.is_emergency else "🚨 Analyzing emergency situation..."):
                    ai_text = generate_medical_advice(
                        feature_row,
                        risk_score, 
                        risk_level,
                        st.session_state.is_emergency,
                        emergency_conditions
                    )

                # Update conversation
                st.session_state.conversation = []
                patient_msg = free_text if free_text.strip() else "Patient provided basic health information"
                st.session_state.conversation.append(("Patient", patient_msg, False))
                
                st.session_state.conversation.append(("Medical AI Assistant", ai_text, st.session_state.is_emergency))
                st.session_state.consultation_done = False
                st.session_state.waiting_for_approval = True
                st.session_state.approval_status = None

                st.rerun()

            # Show conversation if available
            if st.session_state.conversation:
                st.markdown("---")
                st.markdown("## 💬 Consultation Results")
                
                # Risk Score Display
                if st.session_state.prediction_info:
                    rs = st.session_state.prediction_info["risk_score"]
                    rl, risk_class = map_risk_level(rs)
                    
                    st.markdown(f"""
                    <div class='professional-card {risk_class}'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <h3 style='margin: 0; color: #f8fafc;'>Advanced Risk Assessment</h3>
                                <p style='margin: 0.5rem 0 0 0; color: #e2e8f0;'>
                                <strong>Score:</strong> {rs:.1f}/100 | <strong>Level:</strong> {rl}
                                </p>
                                <p style='margin: 0.3rem 0 0 0; color: #94a3b8; font-size: 0.9rem;'>
                                Powered by Ensemble ML with Rule-Based Medical Logic
                                </p>
                            </div>
                            <div style='font-size: 1.5rem; color: #e2e8f0;'>
                                {'🚨' if rs >= 60 else '⚠️' if rs >= 40 else '✅'}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Conversation Display
                st.markdown("#### Consultation Dialogue")
                for speaker, msg, is_emergency in st.session_state.conversation:
                    chat_class = "chat-box chat-emergency" if is_emergency else "chat-box"
                    st.markdown(f"""
                    <div class="{chat_class}">
                        <strong>{speaker}:</strong><br>{msg}
                    </div>
                    """, unsafe_allow_html=True)

                # Summary and Download (only show when approved or if waiting)
                if st.session_state.prediction_info and (st.session_state.approval_status or st.session_state.waiting_for_approval):
                    summary_md = generate_summary_md(
                        st.session_state.conversation,
                        st.session_state.patient_info,
                        st.session_state.prediction_info,
                        st.session_state.is_emergency
                    )
                    
                    with st.expander("📋 Detailed Summary", expanded=True):
                        st.markdown(summary_md)
                    
                    # Download button - only enable when approved
                    if st.session_state.approval_status:
                        st.download_button(
                            label="📥 Download Final Prescription",
                            data=summary_md,
                            file_name=f"prescription_{st.session_state.patient_info.get('name', 'patient')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    else:
                        st.download_button(
                            label="📥 Download Draft Consultation",
                            data=summary_md,
                            file_name=f"consultation_draft_{st.session_state.patient_info.get('name', 'patient')}.md",
                            mime="text/markdown",
                            use_container_width=True,
                            disabled=False
                        )

        # SIDE COLUMN: Information
        with col_side:
            st.markdown("## 📋 Quick Reference")
            
            with st.expander("🆘 Emergency Signs", expanded=True):
                st.markdown("""
                **Seek immediate care for:**
                - Fever ≥ 40°C (104°F)
                - Severe ear pain + fever
                - Chest pain
                - BP > 180/120 or < 90/60
                - Heart rate > 140 or < 40
                - Difficulty breathing
                """)
            
            with st.expander("🌡️ Temperature Guide", expanded=False):
                st.markdown("""
                **Normal:** 36.5-37.5°C (97.7-99.5°F)
                **Fever:** ≥38.0°C (100.4°F)
                **High Fever:** ≥39.5°C (103.1°F)
                **Emergency:** ≥40.0°C (104°F)
                """)
            
            with st.expander("💊 Common Medicines", expanded=False):
                st.markdown("""
                **Fever/Pain:**
                - Paracetamol 650mg
                - Ibuprofen 400mg
                
                **Allergies:**
                - Cetirizine 10mg
                """)

    # =====================================================
    #                 DOCTOR PORTAL
    # =====================================================
    else:
        with col_main:
            st.markdown("## 👨‍⚕️ Doctor Portal")
            
            if st.session_state.waiting_for_approval:
                # Emergency Alert for Doctor
                if st.session_state.is_emergency:
                    st.markdown(f"""
                    <div class='professional-card emergency-card'>
                        <h3 style='color: #f8fafc; margin: 0 0 0.5rem 0;'>🚨 Emergency Case</h3>
                        <p style='margin: 0; color: #f8fafc;'>
                        <strong>Critical conditions:</strong> {', '.join(st.session_state.emergency_conditions)}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.info(f"**Consultation awaiting approval** - Patient: {st.session_state.patient_info.get('name', 'Unknown')}")

                # Patient Summary Card
                if st.session_state.patient_info and st.session_state.prediction_info:
                    st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
                    st.markdown("### Patient Overview")
                    
                    pi = st.session_state.patient_info
                    pred = st.session_state.prediction_info
                    rs = pred.get('risk_score', 0.0)
                    rl, risk_class = map_risk_level(rs)
                    
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.metric("Patient Name", pi.get('name', '-'))
                        st.metric("Age/Gender", f"{pi.get('age', '-')}/{pi.get('gender_display', '-')}")
                    with col_d2:
                        st.metric("Risk Score", f"{rs:.1f}/100")
                        st.metric("Status", "🚨 EMERGENCY" if st.session_state.is_emergency else "🟡 Routine")
                    
                    # Show ML confidence
                    st.markdown("---")
                    st.markdown("#### 🤖 AI Model Insights")
                    st.markdown(f"**Model Used:** {metrics.get('model_name', 'Ensemble')}")
                    st.markdown(f"**Prediction Confidence (R²):** {metrics['r2']:.3f}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)

                # Conversation Review
                st.markdown("### Consultation Review")
                for speaker, msg, is_emergency in st.session_state.conversation:
                    chat_class = "chat-box chat-emergency" if is_emergency else "chat-box"
                    st.markdown(f"""
                    <div class="{chat_class}">
                        <strong>{speaker}:</strong><br>{msg}
                    </div>
                    """, unsafe_allow_html=True)

                # Approval Actions
                st.markdown("### Final Approval")
                col_approve, col_reject = st.columns(2)
                
                with col_approve:
                    if st.button("✅ Approve Consultation", 
                               use_container_width=True,
                               type="primary"):
                        st.session_state.approval_status = True
                        st.session_state.waiting_for_approval = False
                        st.success("✅ Consultation approved and finalized! Patient can now access the prescription.")
                        
                with col_reject:
                    if st.button("❌ Request Offline Visit", 
                               use_container_width=True):
                        st.session_state.approval_status = False
                        st.session_state.waiting_for_approval = False
                        st.warning("Consultation marked for offline evaluation. Patient will be notified.")

            else:
                st.info("No consultations currently waiting for approval.")
                
                # Show previous consultation if available
                if st.session_state.conversation:
                    st.markdown("### Last Consultation Summary")
                    summary_md = generate_summary_md(
                        st.session_state.conversation,
                        st.session_state.patient_info,
                        st.session_state.prediction_info,
                        st.session_state.is_emergency
                    )
                    st.markdown(summary_md)
                    
                    # Show approval status
                    if st.session_state.approval_status is not None:
                        if st.session_state.approval_status:
                            st.success("✅ Last consultation was approved and finalized")
                        else:
                            st.warning("❌ Last consultation required offline visit")
                else:
                    st.markdown("""
                    <div class='professional-card' style='text-align: center; padding: 2rem;'>
                        <h3 style='color: #94a3b8;'>Welcome, Doctor</h3>
                        <p style='color: #94a3b8;'>No consultations in session history.</p>
                    </div>
                    """, unsafe_allow_html=True)

        with col_side:
            st.markdown("## 🎯 Clinical Notes")
            
            with st.expander("📊 Model Details", expanded=True):
                st.markdown(f"""
                **Algorithm:** {metrics.get('model_name', 'Ensemble')}
                **Features:** {len(feature_cols)} parameters
                **R² Score:** {metrics['r2']:.3f}
                **Training Samples:** {metrics['n_train']}
                """)
            
            with st.expander("🔬 Key Metrics", expanded=False):
                st.markdown("""
                **Normal Ranges:**
                - Temp: 36.5-37.5°C
                - BP: 120/80 mmHg
                - HR: 60-100 bpm
                """)

if __name__ == "__main__":
    main()