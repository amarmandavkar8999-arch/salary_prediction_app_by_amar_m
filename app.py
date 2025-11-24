# app.py

import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# --- Configuration & Styling (Attractive as F**k) ---
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, eye-catching look
st.markdown("""
<style>
    /* Main Content Styling */
    .stApp {
        background-color: #0e1117; /* Dark background */
    }
    .st-emotion-cache-18ni2gq { /* Main title container */
        background-color: #0e1117; 
        padding: 20px 0; 
    }
    h1 {
        color: #00bcd4; /* Accent color for the title */
        font-weight: 800;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    .stButton>button {
        background-color: #00bcd4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0097a7;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loading & Caching (The Core) ---
# Use st.cache_resource to load the model only once.
@st.cache_resource
def load_model():
    """Load the serialized machine learning model."""
    try:
        # The file is expected to be in the same directory as app.py
        model_path = Path("sal prediction.pkl")
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            st.stop()
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model. Check scikit-learn version in requirements.txt. Details: {e}")
        st.stop()

model = load_model()

# --- Feature Mapping (Based on Model Snippet) ---
# This dictionary maps user-friendly education levels to numerical values 
# that the model's feature 'Education_Level' is likely expecting.
EDUCATION_MAP = {
    "High School": 1,
    "Bachelor's Degree": 2,
    "Master's Degree": 3,
    "PhD": 4
}


# --- Prediction Function ---
def predict_salary(experience, education_level_num, age):
    """Makes a prediction using the loaded model."""
    # The model expects an array of features in the correct order
    features = np.array([[experience, education_level_num, age]])
    
    # Ensure the model returns a single prediction
    prediction = model.predict(features)[0]
    return prediction

# --- Main App Interface ---

# Header Section
st.title("💸 AI-Powered Salary Estimator")
st.markdown("---")
st.markdown("## Input your professional profile to estimate your annual compensation.")

# 1. User Input Section (Using Columns for a Clean Layout)
st.markdown("### Profile Details")
with st.container():
    # Create two columns: one for key input (Experience), one for other inputs
    col1, col2 = st.columns([1, 1])

    with col1:
        # Input 1: Years of Experience (Using a Slider for interactivity)
        experience = st.slider(
            "💼 **Years of Professional Experience**",
            min_value=0.0, 
            max_value=30.0, 
            value=5.0, 
            step=0.5,
            format="%.1f"
        )
        st.info("Drag the slider to accurately reflect your experience.")

    with col2:
        # Input 2: Education Level (Using a Selectbox for clear categorization)
        education_label = st.selectbox(
            "🎓 **Highest Education Level**",
            options=list(EDUCATION_MAP.keys()),
            index=2 # Default to Master's for a better example start
        )
        education_level_num = EDUCATION_MAP[education_label] # Map to the model's expected number

        # Input 3: Age
        age = st.number_input(
            "🎂 **Age (Years)**",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

st.markdown("---")

# 2. Prediction Trigger
if st.button("🚀 Predict Salary Estimate", key="predict_button"):
    
    # 3. Prediction Logic
    try:
        predicted_salary = predict_salary(experience, education_level_num, age)
        
        # Display the result attractively using st.metric
        st.balloons() # Visual celebration for the prediction!
        
        st.markdown("## 💰 Your Predicted Annual Salary Estimate:")
        st.metric(
            label="Based on your inputs:", 
            value=f"${predicted_salary:,.2f}", 
            delta="Estimate"
        )
        
        st.success("The estimation is complete. This is the projected salary based on the trained model.")

    except Exception as e:
        st.error(f"Prediction Error. Please check your inputs and model compatibility. {e}")

# 4. Sidebar Information
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1593640428585-7b79626b9a2b", width=300) # Placeholder image
    st.title("About This App")
    st.info("This is an end-to-end Machine Learning deployment demo built with **Streamlit**.")
    st.markdown("""
    ---
    ### Model Details
    - **Type**: Linear Regression
    - **Features**: Experience, Education Level, Age
    - **Source**: `sal prediction.pkl`
    """)
