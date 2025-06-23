# 📄 Multiple PDF Q&A Chat (RAG)

Welcome to the **Multiple PDF Q&A Chat (RAG)** project — an AI-powered app that enables you to **ask questions across multiple PDF documents**. Powered by **Retrieval-Augmented Generation (RAG)**, this project processes PDFs into searchable chunks and returns accurate, context-aware responses to your queries.

---
![image](https://github.com/user-attachments/assets/509c4f6a-d8ad-40ab-bdfc-cb634273fa46)

---

##  Key Features
 **Multi-PDF Support** — Upload one or more PDFs at once.  
 **Contextual Answers** — Answers grounded in the contents of your PDFs.  
 **Retrieval-Augmented Generation (RAG)** — Combines vector search with large language models for enhanced accuracy.  
 **User-Friendly UI** — Built with Streamlit for an interactive chat interface.  
 **Fast Semantic Search** — Powered by FAISS for scalable vector similarity lookup.

---

##  How It Works
1.  **Extract Text from PDFs** — PDFs are read and split into manageable chunks.
2.  **Create Embeddings** — Text chunks are transformed into vector embeddings using state-of-the-art models.
3. **Indexing with FAISS** — A FAISS vector index is created to support semantic search.
4.  **Query Processing** — When you ask a question, the most relevant chunks are retrieved.
5.  **Generate Answer** — The retrieved chunks are passed to an LLM to produce a context-aware answer.

---

## 🛠️ Tech Stack
- **Language** — Python 3.x
- **Framework** — Streamlit
- **Core Libraries** — LangChain, PyPDF2, HuggingFace Transformers
- **Embeddings** — `jina-embeddings` or your preferred embedding model
- **Vector Database** — FAISS
- **LLM Backend** — Local or API-driven models (Groq)
  
---
