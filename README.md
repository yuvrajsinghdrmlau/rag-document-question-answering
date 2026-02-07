# rag-document-question-answering
RAG implementation for documents question answering

RAG combines search (retrieval) with generation (LLM) so answers are grounded in your own data, not hallucinations.

# Retrieval-Augmented Generation (RAG) – Document Q&A

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline
that allows users to ask questions from their own documents using
vector search and large language models.

Unlike standard LLMs, this system generates answers **grounded in retrieved
document context**, reducing hallucinations.

---

## 🔍 How the System Works

1. Documents (PDF) are loaded and split into chunks  
2. Each chunk is converted into vector embeddings  
3. FAISS is used for fast similarity search  
4. Relevant chunks are retrieved for a user query  
5. An LLM generates answers using only the retrieved context  

---

## 🛠 Tech Stack

- Python  
- LangChain  
- FAISS (Vector Database)  
- Sentence Transformers (Embeddings)  
- HuggingFace Transformers (LLM)  

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python ingest.py
python query.py
