# Jarvis Local Voice AI

A fully private, local voice assistant that runs entirely on your own machine. No cloud APIs, no subscriptions, no data leaving your computer.

Episode 2 of my series building a private AI setup from scratch.
🎥 Watch the full build: [your YouTube link here]
🎥 Episode 1 (choosing the local LLM): [your YouTube link here]

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
