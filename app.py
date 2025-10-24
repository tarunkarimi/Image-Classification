import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
import plotly.express as px
from streamlit_lottie import st_lottie
import requests


# Page Config

st.set_page_config(
    page_title="Professional CNN Image Classifier 🧠",
    page_icon="🧠",
    layout="wide"
)
st.title("🧠 CNN Image Classification Dashboard")


# Load Lottie Animation

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")
if lottie_ai:
    st_lottie(lottie_ai, height=150, key="ai_animation")


# Load Model (cached)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model

model = load_model()

# CSS Styling
st.markdown("""
    <style>
    /* App background */
    .stApp {
        background-color: #ffffff !important;  /* Overall app white background */
        color: #FF8C00 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* File uploader */
    .stFileUploader>div {
        background-color: #28a745 !important;
        padding: 10px !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
    }

    /* Footer */
    .footer {
        color: #000000 !important;       /* White text */
        text-align: center !important;   /* Center align */
        padding: 12px 0;
        border-radius: 8px !important;
    }

    </style>
""", unsafe_allow_html=True)


# Prediction Function

def predict_image(model, img, target_size=(128,128)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array, verbose=0)[0][0]
    label = "Class1" if pred >= 0.5 else "Class0"
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, confidence


# Session State for History

if "history" not in st.session_state:
    st.session_state.history = []
       

# File Uploader (Centered, Card)

st.markdown('<h3>📂 Upload Images for Prediction</h3>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption=f"🖼 Uploaded: {uploaded_file.name}", use_container_width=True)

        # Predict
        with st.spinner(f"🔍 Predicting {uploaded_file.name}..."):
            start_time = time.time()
            label, confidence = predict_image(model, img)
            end_time = time.time()
        
        # Show prediction in colored card
        st.markdown(f"""
            <div class="card">
                <h4>Prediction for {uploaded_file.name}</h4>
                <p><b>Class:</b> {label}</p>
                <p><b>Confidence:</b> {confidence*100:.2f}%</p>
                <p>⏱ Prediction time: {(end_time - start_time):.2f} sec</p>
            </div>
        """, unsafe_allow_html=True)

        # Save to history
        st.session_state.history.append({
            "Image": uploaded_file.name,
            "Class": label,
            "Confidence": confidence
        })



# Analytics / Dashboard

if st.session_state.history:
    st.markdown("---")
    st.header("📊 Prediction Dashboard")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    # KPIs
    total_predictions = len(df)
    avg_confidence = df['Confidence'].mean()
    class_counts = df['Class'].value_counts().to_dict()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", total_predictions)
    col2.metric("Average Confidence", f"{avg_confidence*100:.2f}%")
    col3.metric("Class1 Count", class_counts.get("Class1",0))
    

    # Bar Chart: Confidence per Image
    fig_bar = px.bar(df, x="Image", y="Confidence", color="Class",
                     text=df["Confidence"].apply(lambda x: f"{x*100:.1f}%"),
                     color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Pie Chart: Class Distribution (fixed)
    class_summary = df['Class'].value_counts().reset_index()
    class_summary.columns = ['Class', 'Count']
    fig_pie = px.pie(class_summary, names='Class', values='Count',
                     color='Class', color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Download CSV
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="📥 Download Prediction History CSV",
        data=csv,
        file_name='prediction_history.csv',
        mime='text/csv'
    )


# Footer

st.markdown(""" <div class='footer'> 👨‍💻 Developed by Tarun Teja Karimi | Powered by TensorFlow & Streamlit | Dashboard Version </div> """, unsafe_allow_html=True)



