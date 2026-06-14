import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the data and model
songs = pd.read_csv("deployment_songs.csv")
embeddings = np.load("deployment_embeddings.npy")
model = joblib.load("genre_model.pkl")

st.title("🎵 Spotify Genre Predictor")

song_name = st.selectbox("Choose a Song", songs["name"])

if st.button("Predict Genre"):
    row_index = songs[songs["name"] == song_name].index[0]
    embedding = embeddings[row_index].reshape(1, -1)
    
    prediction = model.predict(embedding)[0]
    actual = songs.iloc[row_index]["genre"]
    artist = songs.iloc[row_index]["artists"]

    st.write(f"Artist: {artist}")
    st.write(f"Actual Genre: {actual}")

    st.success(f"Predicted Genre: {prediction}")