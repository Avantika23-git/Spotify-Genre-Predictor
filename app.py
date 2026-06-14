import streamlit as st
import requests

st.set_page_config(
    page_title="Spotify Genre Predictor",
    page_icon="🎵"
)

st.title("🎵 Spotify Genre Predictor")

st.write(
    "Upload an MP3 song and predict its genre using AI."
)

uploaded_file = st.file_uploader(
    "Choose an MP3 file",
    type=["mp3"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Predict Genre"):

        with st.spinner("Analyzing song..."):

            files = {
                "file": uploaded_file
            }

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files
            )

            result = response.json()

            st.success(
                f"Predicted Genre: {result['predicted_genre']}"
            )