import tensorflow_hub as hub
import librosa
import numpy as np
import joblib

print("Loading VGGish...")

vggish = hub.load(
    "https://tfhub.dev/google/vggish/1"
)

print("Loading Random Forest...")

rf = joblib.load("genre_model.pkl")

audio, sr = librosa.load(
    "sample.mp3",
    sr=16000,
    mono=True
)

print("Generating Embeddings...")

embedding = vggish(audio)

song_embedding = embedding.numpy().mean(axis=0)

print("Embedding Shape:", song_embedding.shape)

prediction = rf.predict(
    song_embedding.reshape(1, -1)
)

print("\nPredicted Genre:", prediction[0])