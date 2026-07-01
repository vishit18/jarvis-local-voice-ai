import urllib.request
import os

BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"

files = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]

for filename in files:
    if os.path.exists(filename):
        print(f"{filename} already exists, skipping.")
        continue
    print(f"Downloading {filename}... (this may take a few minutes)")
    urllib.request.urlretrieve(BASE_URL + filename, filename)
    print(f"Done: {filename}")

print("All files ready.")