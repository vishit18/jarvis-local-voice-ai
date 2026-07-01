import sounddevice as sd
import numpy as np
import requests
import soundfile as sf
import os
import tempfile
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
import re

def clean_text_for_speech(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub('', text).strip()

# ── Load models ──────────────────────────────────────────
print("Loading Whisper...")
stt = WhisperModel("small", device="cuda", compute_type="float16")
print("Whisper ready.")

print("Loading Kokoro...")
tts = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
print("Kokoro ready.")

print("\n✅ All systems ready. Press Enter to speak, Ctrl+C to quit.\n")

# ── Functions ─────────────────────────────────────────────
def listen(max_seconds=15, silence_threshold=0.005, silence_duration=2.0):
    print("🎤 Listening... (speak now)")
    sample_rate = 16000
    chunk_duration = 0.1
    chunk_samples = int(sample_rate * chunk_duration)
    
    audio_chunks = []
    silence_chunks = 0
    max_silence_chunks = int(silence_duration / chunk_duration)
    started_speaking = False
    
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32')
    stream.start()
    
    total_chunks = int(max_seconds / chunk_duration)
    
    for _ in range(total_chunks):
        chunk, _ = stream.read(chunk_samples)
        audio_chunks.append(chunk)
        
        volume = np.abs(chunk).mean()
        
        if volume > silence_threshold:
            started_speaking = True
            silence_chunks = 0
        elif started_speaking:
            silence_chunks += 1
            if silence_chunks >= max_silence_chunks:
                break
    
    stream.stop()
    stream.close()
    
    audio = np.concatenate(audio_chunks, axis=0)
    return audio.flatten()

def transcribe(audio):
    segments, _ = stt.transcribe(audio, beam_size=5)
    text = " ".join([s.text for s in segments]).strip()
    print(f"\n💬 You said: {text}")
    return text

def ask_llm(text):
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": "qwen/qwen3-vl-8b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant running locally on the user's own laptop with no internet connection. Keep all responses to two or three sentences maximum. Be conversational and natural."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                "max_tokens": 100
            },
            timeout=30
        )
        reply = response.json()['choices'][0]['message']['content']
        print(f"🤖 AI: {reply}\n")
        return reply
    except Exception as e:
        return f"Error connecting to LM Studio: {str(e)}"

def speak(text):
    text = clean_text_for_speech(text)
    samples, sample_rate = tts.create(
        text,
        voice="bm_daniel",
        speed=0.97,
        lang="en-us"
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, samples, sample_rate)

    import subprocess
    subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{tmp_path}").PlaySync()'])

    os.remove(tmp_path)

# ── Main loop ─────────────────────────────────────────────
while True:
    input("Press Enter to speak...")
    audio = listen(max_seconds=15, silence_threshold=0.005, silence_duration=2.0)
    text = transcribe(audio)
    if text:
        reply = ask_llm(text)
        speak(reply)
    else:
        print("Didn't catch that. Try again.")