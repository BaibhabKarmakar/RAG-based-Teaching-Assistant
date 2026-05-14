# 🎓 RAG-Based AI Teaching Assistant

An AI-powered teaching assistant built using **Retrieval-Augmented Generation (RAG)** architecture. It takes a set of lecture videos as input and allows users to ask questions about the course content — returning precise answers with **video titles and timestamps**.

---

## 🧠 How It Works

```
Videos → Audio → Text Chunks → Embeddings → Vector Search → LLM → Answer
```

### Pipeline Overview

| Step | Description | Tools Used |
|------|-------------|------------|
| 1 | Convert videos to audio | `ffmpeg` |
| 2 | Transcribe audio into text chunks | `faster-whisper` (medium, int8) |
| 3 | Convert text chunks to embeddings | `bge-m3` via Ollama |
| 4 | Store embeddings as a DataFrame | `pandas`, `joblib` |
| 5 | Find relevant chunks via cosine similarity | `scikit-learn` |
| 6 | Generate answer using LLM | `deepseek-reasoner` via DeepSeek API |

---

## 📁 Project Structure

```
RAG_based_AI_teaching_assistant/
│
├── Videos/                  # ← Your lecture videos go here (not included)
├── audios/                  # ← Generated audio files (not included)
├── jsons/                   # ← Generated transcript chunks (not included)
│
├── video_to_mp3.py          # Step 1: Convert videos to mp3
├── mp3_to_json_chunks.py    # Step 2: Transcribe and chunk audio
├── preprocessing_json.py    # Step 3: Generate and store embeddings
├── process_incoming.py      # Step 4 & 5: Query and get answers
│
├── embeddings.joblib        # ← Generated embeddings (not included)
├── .env                     # ← Your API key (not included)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

Make sure you have these installed:

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) — for video to audio conversion
- [Ollama](https://ollama.com/download) — for running the embedding model locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RAG_based_AI_teaching_assistant.git
cd RAG_based_AI_teaching_assistant
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv RAGenv

# Windows
.\RAGenv\Scripts\activate

# Mac/Linux
source RAGenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Embedding Model via Ollama

```bash
ollama pull bge-m3
```

### 5. Set Up Your API Key

Create a `.env` file in the root directory:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

Get your API key from 👉 [platform.deepseek.com](https://platform.deepseek.com/api_keys)

---

## 🚀 Running the Project on Your Own Data

### Step 1 — Add Your Videos

Place all your lecture video files inside the `Videos/` folder.

```
Videos/
├── lecture1.mp4
├── lecture2.mp4
└── ...
```

### Step 2 — Convert Videos to Audio

```bash
python video_to_mp3.py
```

Converts all video files in `Videos/` to `.mp3` format using `ffmpeg`.

### Step 3 — Transcribe Audio to Text Chunks

```bash
python mp3_to_json_chunks.py
```

Uses `faster-whisper` to transcribe audio and splits them into chunks with timestamps. Saves output as JSON files in `jsons/`.

> **Note:** This project uses `faster-whisper` with `medium` model, `int8` quantization, and `beam_size=1` for lower RAM usage. You can upgrade to `large` model if your system supports it.

### Step 4 — Generate Embeddings

```bash
python preprocessing_json.py
```

Converts all text chunks to vector embeddings using `bge-m3` (via Ollama) and saves them as `embeddings.joblib`.

> **Note:** Make sure Ollama is running before this step:
> ```bash
> ollama serve
> ```

### Step 5 — Ask Questions!

```bash
python process_incoming.py
```

Enter your question and the assistant will find the most relevant video segments and return a detailed answer with video names and timestamps.

---

## 💬 Example Output

```
Ask a Question: What is interpretability?

--- Answer ---
Interpretability is covered extensively in video 25 "Interpretability",
starting at 00:00:17.0 where the instructor introduces the topic...

It is also mentioned in:
- "Risk_Stratification_part1" (video 4) at 00:31:51.0
- "Fairness" (video 23) at 00:08:06.0
```

---

## 🔧 Technical Details

- **Embedding Model:** `bge-m3` — a powerful multilingual embedding model running locally via Ollama (free, no API needed)
- **LLM:** `deepseek-reasoner` — accessed via DeepSeek API (requires balance, very cheap ~$0.00154/request)
- **Similarity Search:** Cosine similarity using `scikit-learn`
- **Storage:** `joblib` pickle for embeddings DataFrame

---

## 📦 Requirements

```txt
scikit-learn
pandas
joblib
requests
openai
python-dotenv
faster-whisper
```

Install all at once:

```bash
pip install scikit-learn pandas joblib requests openai python-dotenv faster-whisper
```

---

## ⚠️ Important Notes

- Keep Ollama running in the background while using the project (`ollama serve`)
- Never commit your `.env` file — it contains your secret API key
- The `Videos/`, `jsons/`, and `embeddings.joblib` are excluded from the repo due to size — you must generate them from your own videos

---

## 🙋 Author

**Baibhab Karmakar**
B.Tech IT, University of Calcutta
[GitHub](https://github.com/BaibhabKarmakar) • [LinkedIn](https://linkedin.com/in/your-profile)