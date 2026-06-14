import librosa

audio, sr = librosa.load(
    "sample.mp3",
    sr=16000
)

print("Shape:", audio.shape)
print("Sample Rate:", sr)
print("Duration:", len(audio)/sr)