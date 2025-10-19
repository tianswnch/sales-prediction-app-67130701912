# app.py
import streamlit as st
import pandas as pd
import joblib
import glob
import os
from io import BytesIO

st.set_page_config(page_title="Advertising Sales Predictor", layout="centered")

st.title("Advertising → Sales (Linear Regression)")
st.write("Load a trained `model-reg-67130701912.pkl` and predict sales from three inputs.")

def load_model_from_bytesio(f):
    # joblib.load accepts file-like objects (BytesIO)
    f.seek(0)
    return joblib.load(f)

def find_latest_model():
    matches = glob.glob("model-reg-67130701912.pkl")
    if not matches:
        return None
    latest = max(matches, key=os.path.getmtime)
    return latest

# Sidebar: model selection / upload
st.sidebar.header("Model")
uploaded_model = st.sidebar.file_uploader("Upload model-reg-67130701912.pkl", type=["pkl"])

use_latest = False
if uploaded_model is None:
    latest = find_latest_model()
    if latest:
        st.sidebar.write(f"Auto-detected model: `{latest}`")
        use_latest = st.sidebar.button("Use auto-detected model")
    else:
        st.sidebar.write("No local `model-reg-67130701912.pkl` found. Please upload a model.")

model = None
if uploaded_model:
    try:
        model = load_model_from_bytesio(uploaded_model)
        st.sidebar.success("Model loaded from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded model: {e}")
elif use_latest:
    try:
        model = joblib.load(latest)
        st.sidebar.success(f"Loaded model: {latest}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model '{latest}': {e}")

st.header("Input features")
# default values 50,50,50
youtube = st.number_input("YouTube (budget/score)", value=50, min_value=0)
tiktok = st.number_input("TikTok (budget/score)", value=50, min_value=0)
instagram = st.number_input("Instagram (budget/score)", value=50, min_value=0)

if st.button("Predict"):

    X_new = pd.DataFrame([[youtube, tiktok, instagram]],
                         columns=["youtube", "tiktok", "instagram"])
    st.write("Input row:")
    st.write(X_new)

    if model is None:
        st.error("No model loaded. Upload a `model-reg-67130701912.pkl` file or use a local one named `model-reg-67130701912.pkl`.")
    else:
        try:
            y_pred = model.predict(X_new)
            st.success(f"Predicted sales: {float(y_pred[0]):.4f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.write("Notes:")
st.write("- Make sure the trained model expects features in this order and uses these column names.")
st.write("- If your training included scaling/encoding pipelines, be sure to save & load the full pipeline (e.g., a `Pipeline` object).")
