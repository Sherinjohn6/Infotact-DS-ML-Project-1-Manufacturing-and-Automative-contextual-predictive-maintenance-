import streamlit as st
import pandas as pd
import pickle
import joblib
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI4I Predictive Maintenance",
    layout="wide"
)

# ===============================
# CUSTOM BLUE THEME (YOUR COLORS)
# ===============================
st.markdown("""
<style>

.stApp {
    background-color: #F8FAFC;
    color: #1F2937;
}

h1, h2, h3 {
    color: #2563EB;
}

/* Button */
.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    width: 100%;
    height: 3em;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #1D4ED8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #E5E7EB;
}

/* Success */
div[data-testid="stAlert-success"] {
    background-color: #22C55E;
    color: white;
}

/* Error */
div[data-testid="stAlert-error"] {
    background-color: #EF4444;
    color: white;
}

/* Warning */
div[data-testid="stAlert-warning"] {
    background-color: #F59E0B;
    color: black;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = "final_lightgbm_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except:
    model = pickle.load(open(MODEL_PATH, "rb"))

# ===============================
# LOAD FEATURE NAMES
# ===============================
feature_names = pickle.load(open("feature_names.pkl", "rb"))

# ===============================
# TITLE
# ===============================
st.title("🔧 AI4I Predictive Maintenance Dashboard")
st.write("Enter all required feature values below")

# ===============================
# INPUT SECTION (REAL FEATURES)
# ===============================
st.sidebar.header("⚙️ Input Features")

input_data = {}

for feature in feature_names:

    # If categorical feature (AI4I dataset example)
    if feature.lower() in ["type_l", "type_m", "type_h"]:
        input_data[feature] = st.sidebar.selectbox(
            f"{feature}",
            [0, 1]
        )
    else:
        input_data[feature] = st.sidebar.number_input(
            f"{feature}",
            value=0.0
        )

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict Failure"):

    input_df = pd.DataFrame([input_data])

    # ensure correct order
    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ MACHINE FAILURE PREDICTED")
        st.write("Immediate maintenance required.")
    else:
        st.success("✅ NO FAILURE DETECTED")
        st.write("Machine is operating normally")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("💙 AI4I Predictive Maintenance | LightGBM + Streamlit")