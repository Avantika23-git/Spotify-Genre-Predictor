from fastapi import FastAPI, UploadFile, File
import tensorflow_hub as hub
import librosa
import numpy as np
import joblib
import tempfile

app = FastAPI()

print("Loading VGGish...")
vggish = hub.load("https://tfhub.dev/google/vggish/1")

print("Loading Model...")
rf = joblib.load("genre_model.pkl")


@app.get("/")
def home():
    return {"message": "Genre API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    audio, sr = librosa.load(
        temp_path,
        sr=16000,
        mono=True
    )

    embedding = vggish(audio)

    song_embedding = embedding.numpy().mean(axis=0)

    prediction = rf.predict(
        song_embedding.reshape(1, -1)
    )

    return {
        "predicted_genre": prediction[0]
    }