# Jarvis Local Voice AI

A fully private, local voice assistant that runs entirely on your own machine. No cloud APIs, no subscriptions, no data leaving your computer.

Part of my series building a private AI setup from scratch.

🎥 Episode 1 (choosing the local LLM): [https://www.youtube.com/watch?v=in-Y2Sg_ZJ0&t=239s]

🎥 Episode 2 (giving it a voice): [https://www.youtube.com/watch?v=mn74jKcBPo8]

🎥 Episode 3 (giving it memory): []
## How it works

1. **faster-whisper** listens to your voice and converts it to text
2. **LM Studio** (running Qwen3 locally) generates a response
3. **Kokoro ONNX** converts the response back into natural speech

Everything runs on your own hardware. The LLM never makes an outbound connection.

## Requirements

- Python 3.10 (newer versions may have dependency issues with some packages)
- NVIDIA GPU recommended for faster Whisper transcription
- [LM Studio](https://lmstudio.ai) installed and running locally with a model loaded

## Setup

1. Clone this repo
   git clone https://github.com/vishit18/jarvis-local-voice-ai.git
   cd jarvis-local-voice-ai

2. Create a virtual environment with Python 3.10
   py -3.10 -m venv venv
   venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Download the Kokoro model files
   python download_kokoro.py

5. Open LM Studio, load a model, and start the local server (default port 1234)

6. Run Jarvis
   python jarvis.py

## Notes

- Update the `model` field in `jarvis.py` to match whatever model identifier LM Studio shows for your loaded model
- Default voice is `bm_lewis` — Kokoro has 26+ voices available
- This is a learning project built for a YouTube series, not a polished product.

## Memory (Episode 3)

Jarvis now remembers past conversations using ChromaDB, a local vector database. Every exchange gets saved, and relevant past context gets pulled in automatically before each response — no manual copy-pasting of chat history required.

Run `test_memory.py` first to see the memory system working standalone before running the full `jarvis.py` loop.

Note: if you're running Whisper and the LLM on the same GPU, adding the memory layer can push VRAM usage past what a single GPU comfortably handles. If you hit slow responses or timeouts, try moving Whisper to CPU:
```python
stt = WhisperModel("small", device="cpu", compute_type="int8")
```

## License

MIT — do whatever you want with this.
