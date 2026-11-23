# AI Research Assistant

Ask questions about your PDF notes using Retrieval-Augmented Generation (RAG).

**Pipeline:** PDF → text extraction (pypdf) → chunking (1000 chars, 200 overlap) →
embeddings (`all-MiniLM-L6-v2`) → Chroma vector store → top-k retrieval → answer from a Groq LLM with page citations.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, smaller install
pip install -r requirements.txt
cp .env.example .env   # then put your GROQ_API_KEY in .env
```

## Usage

Put your PDF at `data/pdfs/Notes.pdf`, then from the project root:

```bash
python app/main.py              # first run builds the index in chroma_db/ (takes a few minutes)
python app/main.py my_paper.pdf # any PDF; each gets its own index, built on first use
python app/main.py --rebuild    # force re-indexing
```

## Project structure

| File | Purpose |
|------|---------|
| `app/main.py` | Interactive CLI chat |
| `app/rag_pipeline.py` | `AIResearchAssistant` class: ingestion, retrieval, answering |
| `app/chunker.py` | Learning step: naive fixed-size chunking |
| `app/text_splitter.py` | Learning step: recursive "smart" chunking |
