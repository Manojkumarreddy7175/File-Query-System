# File Query System — RAG Pipeline

A lightweight **Retrieval-Augmented Generation (RAG)** app that lets you upload any document and ask questions about it in plain English. Powered by **Groq** for blazing-fast LLM inference and **FAISS** for local vector search.

---

## What Is It?

Most LLMs don't know the contents of *your* files. This app solves that by:

1. Breaking your document into small chunks
2. Converting those chunks into semantic vectors (embeddings) stored locally
3. When you ask a question, finding the most relevant chunks
4. Sending only those chunks + your question to the LLM — so the answer is grounded in your document

This approach is called **RAG** (Retrieval-Augmented Generation) and it works without fine-tuning or uploading your whole file to an external API.

---

## Features

- Upload **PDF**, **DOCX**, or **TXT** files
- Ask unlimited questions in a clean chat interface
- Answers are grounded in your document — no hallucination from unrelated knowledge
- Expandable **source passages** shown alongside every answer
- Embeddings run **100% locally** (no data sent for indexing)
- Choose from multiple **Groq models** (Llama 3, Mixtral, Gemma)

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Document                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ Extract text
                           ▼
                   ┌───────────────┐
                   │  Text Chunks  │  (800 chars, 100 overlap)
                   └───────┬───────┘
                           │ Embed locally
                           ▼
                   ┌───────────────┐
                   │  FAISS Index  │  (all-MiniLM-L6-v2)
                   └───────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                                 │
   Your Question                    Top 4 chunks retrieved
          │                                 │
          └────────────────┬────────────────┘
                           │ Send to Groq LLM
                           ▼
                   ┌───────────────┐
                   │    Answer     │
                   └───────────────┘
```

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (Llama 3.3 70B / Mixtral / Gemma) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local) |
| Vector Store | FAISS (in-memory) |
| Document Parsing | PyPDF, python-docx |

---

## Setup & Installation

### Prerequisites

- Python **3.10 – 3.12** (3.14+ not supported by dependencies)
- A free **Groq API key** → [console.groq.com](https://console.groq.com)

### 1. Clone the repo

```bash
git clone https://github.com/Manojkumarreddy7175/File-Query-System.git
cd File-Query-System
```

### 2. Create a virtual environment

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run will download the `all-MiniLM-L6-v2` embedding model (~90 MB) and cache it automatically.

### 4. Add your Groq API key

Copy the example env file and fill in your key:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Then edit `.env`:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

> Alternatively, you can paste the key directly in the app's sidebar — no `.env` file needed.

### 5. Run the app

```bash
# Windows
.venv\Scripts\streamlit.exe run app.py

# macOS / Linux
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## How to Use

1. **Enter your Groq API key** in the left sidebar (auto-loaded if `.env` is set)
2. **Choose a model** from the dropdown (Llama 3.3 70B recommended for best answers)
3. **Upload a file** — PDF, DOCX, or TXT
4. Click **⚡ Process Document** and wait for the index to build (usually 5–15 seconds)
5. **Type your question** in the chat box and hit Enter
6. The answer appears in the chat — click **📚 Source passages** to see exactly which parts of the document were used

---

## Project Structure

```
File-Query-System/
├── app.py              # Streamlit UI — chat interface, sidebar, file upload
├── rag_pipeline.py     # Core logic — text extraction, embedding, retrieval, answer generation
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── .gitignore
```

---

## Models Available

| Model | Best For |
|---|---|
| `llama-3.3-70b-versatile` | Best quality answers (recommended) |
| `llama3-8b-8192` | Fast responses, lower latency |
| `mixtral-8x7b-32768` | Long documents (32k context) |
| `gemma2-9b-it` | Lightweight, efficient |

---

## Privacy

- Your document is processed **entirely on your machine** for embedding
- Only the **query + top 4 relevant text chunks** are sent to Groq's API
- Nothing is stored between sessions — the index is rebuilt on each upload

---

## License

MIT
