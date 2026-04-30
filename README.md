# RAG-Powered Search Engine
## Q&A with Retrieval-Augmented Generation

**Final Capstone Project | AI Application Development**

---

## Overview

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline that enables
an LLM to answer questions grounded in a custom knowledge base — eliminating hallucinations and
enabling domain-specific Q&A without fine-tuning.

### Architecture

```
INGESTION PIPELINE (once):
  Documents -> Text Extraction -> RecursiveCharacterTextSplitter
            -> Embeddings (all-MiniLM-L6-v2) -> FAISS Index

QUERY PIPELINE (per request):
  User Query -> Embed Query -> FAISS Top-K Similarity Search
             -> Retrieved Chunks -> RAG Prompt Template
             -> LLM (flan-t5-base / gpt-3.5-turbo) -> Grounded Answer + Sources
```

---

## Project Structure

```
Final_Capstone_RAG/
├── project.ipynb          # Main notebook — full end-to-end implementation
├── app.py                 # Streamlit web application for deployment
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── faiss_rag_index/       # Persisted FAISS vector store (generated at runtime)
    ├── index.faiss
    └── index.pkl
```

---

## Deliverables

| File | Description |
|---|---|
| `project.ipynb` | End-to-end RAG implementation with problem statement, code, testing, evaluation, and results |
| `app.py` | Interactive Streamlit web app — type questions, get grounded answers |
| `requirements.txt` | All Python dependencies with version constraints |
| `README.md` | Project documentation (this file) |

---

## Setup and Installation

### 1. Clone / Download the project

```bash
unzip Final_Capstone_Project.zip
cd Final_Capstone_RAG
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
``` 

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Jupyter Notebook

```bash
jupyter notebook project.ipynb
```

Run cells top-to-bottom. The notebook will:
1. Install packages
2. Build the knowledge base
3. Chunk documents
4. Create embeddings and FAISS index
5. Assemble the RAG chain
6. Run 8 test questions with evaluation
7. Display ROUGE-L scores, retrieval accuracy, latency stats, and visualisation charts

---

## Running the Streamlit App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

Features:
- Type any question about AI/ML, NLP, Transformers, or RAG
- Click sample question buttons for quick demos
- Expand retrieved chunks to see exactly what context the LLM used
- Browse the full knowledge base
- Toggle between FREE (HuggingFace) and OpenAI mode in the sidebar

---

## Using OpenAI (Optional)

By default the project runs **100% free** using HuggingFace models.

To use OpenAI GPT-3.5-turbo:

**In the notebook:**
```python
USE_OPENAI     = True
OPENAI_API_KEY = "sk-your-key-here"
```

**In the Streamlit app:**
Toggle "Use OpenAI GPT-3.5" in the sidebar and paste your API key.

---

## Knowledge Base

The default knowledge base covers 4 documents (~1,800 words each):

| Document | Topic |
|---|---|
| AI and ML Fundamentals | Supervised/unsupervised/RL, deep learning, overfitting, metrics |
| NLP Techniques | Word embeddings, attention, BERT, GPT, evaluation |
| Transformer Architecture | Attention formula, encoder/decoder, RAG origin, FAISS, chunking |
| Production RAG Systems | Ingestion, retrieval strategies, prompt engineering, RAGAS, latency |

### Adding Your Own Documents

**Option A — Plain text in notebook:**
Add entries to the `DOCUMENTS` dict in Section 4.

**Option B — PDF files:**
Use the `load_pdfs_from_folder()` helper in Section 4b:
```python
pdf_docs = load_pdfs_from_folder("./my_documents")
```

---

## Evaluation Results

| Metric | Value |
|---|---|
| Retrieval Accuracy | >= 85% (correct source retrieved) |
| Average ROUGE-L Score | ~0.35-0.55 |
| Out-of-Scope Handling | PASS (system refuses unknown questions) |
| Average Latency (free mode) | ~2-5s per query |
| Total Chunks Indexed | ~35-40 chunks from 4 documents |

---

## Tech Stack

| Component | Technology |
|---|---|
| RAG Framework | LangChain |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS (Facebook AI Similarity Search) |
| Free LLM | google/flan-t5-base (HuggingFace) |
| OpenAI LLM | gpt-3.5-turbo |
| Web App | Streamlit |
| Evaluation | ROUGE-Score, Retrieval Recall |
| Visualisation | Matplotlib |

---

## Future Improvements

- Domain-specific embedding fine-tuning (+15-25% retrieval accuracy)
- BM25 hybrid retrieval for keyword matching
- Cross-encoder re-ranking of top-k results
- RAGAS automated evaluation pipeline
- Parent Document Retrieval for better coherence
- LLM response streaming for lower perceived latency

---

*Final Capstone Project — AI Application Development*
