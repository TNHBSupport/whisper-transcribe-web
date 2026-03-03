# 🎙️ whisper-transcribe-web

A local web interface for transcribing video/audio files using [OpenAI Whisper](https://github.com/openai/whisper), with optional AI summarization via [Ollama](https://ollama.com). Built with [Gradio](https://gradio.app).

---

## ✨ Features

- 🎬 **Upload any video or audio file** and transcribe it locally — no cloud, no API keys
- 📄 **Outputs structured JSON + plain TXT** transcripts (Format C compatible)
- 🧠 **Optional AI summarization** via a locally running Ollama model (outputs `.md` + `.json`)
- 🔢 **Auto episode numbering** — inferred from filename (e.g. `01_interview.mp4` → Episode 1)
- 🛠️ **Manual episode override** via the UI
- ⚙️ **Model & device selection** — choose Whisper model size and CPU/CUDA
- 🌐 **Runs entirely on your machine** — privacy-first, no data leaves your computer

---

## 📦 Requirements

### System

- Python 3.10+
- `ffmpeg` installed and available on your `PATH`
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: [Download from ffmpeg.org](https://ffmpeg.org/download.html)

### Python Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```
gradio>=4.0.0
openai-whisper>=20231117
torch>=2.0.0
requests
```

> **GPU users:** Install a CUDA-compatible version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

### Optional: Ollama (for summarization)

To enable the **summarization feature**, install and run [Ollama](https://ollama.com) locally:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (e.g. LLaMA 3.1 8B)
ollama pull llama3.1:8b

# Start the Ollama server
ollama serve
```

---

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/whisper-transcribe-web.git
cd whisper-transcribe-web

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python transcribe_web.py
```

Then open your browser at `http://localhost:7860`.

### UI Options

| Option | Description |
|--------|-------------|
| **Upload video/audio** | Supports `.mp4`, `.mkv`, `.mov`, `.mp3`, `.wav`, `.m4a`, etc. |
| **Whisper model** | `tiny` → fastest, `large-v3` → most accurate |
| **Device** | `cpu` (default) or `cuda` (requires NVIDIA GPU + CUDA) |
| **Episode override** | Set manually or leave `0` to auto-detect from filename |
| **Generate summary** | Requires Ollama running locally |
| **Ollama model** | Default: `llama3.1:8b` |
| **Ollama URL** | Default: `http://localhost:11434/api/generate` |
| **Chunk seconds** | Split long audio into chunks for summarization (120–1800s) |

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `{ep}_transcript.json` | Structured transcript (Format C) with timestamps |
| `{ep}_transcript.txt` | Plain text transcript |
| `{ep}_summary.json` | Structured summary (Format A) — if summarization enabled |
| `{ep}_summary.md` | Markdown summary — if summarization enabled |

---

## 🗂️ Project Structure

```
whisper-transcribe-web/
├── transcribe_web.py       # Gradio web UI (this file)
├── transcribe_upgrade.py   # Core transcription & summarization logic
├── requirements.txt
└── README.md
```

---

## 🔧 Troubleshooting

**`ffmpeg not found`** — Make sure `ffmpeg` is installed and on your system PATH.

**Slow transcription** — Use a smaller Whisper model (`tiny` or `base`) or enable CUDA if you have a GPU.

**Ollama connection refused** — Make sure Ollama is running (`ollama serve`) before enabling summarization.

**CUDA out of memory** — Switch device to `cpu` or use a smaller Whisper model.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
