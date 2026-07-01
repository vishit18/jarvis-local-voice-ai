import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

print("Loading Whisper... (first run downloads ~500MB)")
model = WhisperModel("small", device="cuda", compute_type="float16")
print("Whisper ready.")

input("Press Enter then speak clearly for 5 seconds...")
print("Recording...")
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
print("Processing...")

segments, _ = model.transcribe(audio.flatten(), beam_size=5)
text = " ".join([s.text for s in segments])
print(f"You said: {text}")
