import sounddevice as sd
import numpy as np
import requests
import soundfile as sf
import os
import tempfile
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
import re
from chromadb.utils import embedding_functions


import chromadb

print("Loading memory...")
memory_client = chromadb.PersistentClient(path="./jarvis_memory")

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)

memory_collection = memory_client.get_or_create_collection(
    "conversations",
    embedding_function=embed_fn
)
print("Memory ready.")

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
stt = WhisperModel("small", device="cpu", compute_type="int8")
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
    memory_context = recall(text)
    
    system_prompt = "You are a helpful assistant running locally on the user's own laptop with no internet connection. Keep all responses to two or three sentences maximum. Be conversational and natural. Do not use emojis."
    
    if memory_context:
        system_prompt += f"\n\nRelevant context from past conversations:\n{memory_context}"
    
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": "qwen/qwen3-vl-8b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "max_tokens": 60
            },
            timeout=180
        )
        reply = response.json()['choices'][0]['message']['content']
        print(f"\n🤖 AI: {reply}\n")
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

def remember(text, role):
    memory_collection.add(
        documents=[text],
        ids=[f"{role}_{memory_collection.count()}"],
        metadatas=[{"role": role}]
    )

def recall(query, n=3):
    if memory_collection.count() == 0:
        return ""
    results = memory_collection.query(query_texts=[query], n_results=min(n, memory_collection.count()))
    return "\n".join(results['documents'][0])

# ── Main loop ─────────────────────────────────────────────
while True:
    input("Press Enter to speak...")
    audio = listen(max_seconds=20, silence_threshold=0.005, silence_duration=2.0)
    text = transcribe(audio)
    if text and len(text) > 2:
        remember(text, "user")
        reply = ask_llm(text)
        remember(reply, "assistant")
        speak(reply)
    else:
        print("Didn't catch that. Try again.\n")