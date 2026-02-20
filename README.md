# 🧠 Basic RAG System — PDF Question Answering with Gemini

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask natural language questions about PDF documents. Built with Sentence Transformers, Chromadb, and Google Gemini.

---

## 📁 Project Structure

```
basic_rag_system/
│
├── main.py                  # Entry point
├── config.py                # Configuration settings
├── requirements.txt         # Dependencies
│
├── tools/
│   ├── __init__.py
│   ├── document_loader.py   # Loads PDF documents
│   ├── chunker.py           # Splits text into chunks
│   ├── embedder.py          # Creates embeddings & vector DB
│   ├── retriever.py         # Retrieves relevant chunks
│   └── generator.py         # Generates answers via Gemini
│
├── data/
│   └── raw/
│       └── Beta_vae.pdf     # Input PDF
│
└── vector_db/               # Stored embeddings (auto-generated)
```

---

## ⚙️ How It Works

```
PDF → Chunk → Embed → Store → Retrieve → Generate
```

1. **Load** — Read the PDF document
2. **Chunk** — Split text into overlapping segments
3. **Embed** — Convert chunks to vector embeddings
4. **Store** — Save embeddings to a FAISS vector database
5. **Retrieve** — Find the top-k most relevant chunks for a query
6. **Generate** — Send context + query to Gemini and return an answer

---

## 📦 Installation

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirement.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

> Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## ▶️ Running the Project

Run from the **project root** (the folder *containing* `basic_rag_system/`):

```bash
python -m basic_rag_system.main
```

> ⚠️ Do **not** run `python main.py` directly — relative imports will fail.

---

## 🧪 Example Query

```
What is the topic of the document?
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Embeddings | Sentence Transformers |
| Vector Store | Chromadb |
| LLM | Google Gemini API |
| PDF Parsing | PyPDF |
| Environment | python-dotenv |


## 👨‍💻 Author

**Vishva MV**  
MSc Data Science — University of Hertfordshire
