import tensorflow_hub as hub

print("Loading VGGish...")

model = hub.load(
    "https://tfhub.dev/google/vggish/1"
)

print("Loaded Successfully!")