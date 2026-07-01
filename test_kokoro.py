from kokoro_onnx import Kokoro
import soundfile as sf
import os
import subprocess

print("Loading Kokoro...")
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

text = "Hello. I am running entirely on your laptop. No internet connection required."

print("Generating speech...")
samples, sample_rate = kokoro.create(
    text,
    voice="af_heart",
    speed=1.0,
    lang="en-us"
)

sf.write("test_output.wav", samples, sample_rate)
print("Done. Playing audio...")

subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "test_output.wav").PlaySync()'])