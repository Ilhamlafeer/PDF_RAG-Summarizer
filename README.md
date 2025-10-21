# 📚 PDF RAG Summarizer

**PDF RAG Summarizer** is an AI-powered tool that retrieves and summarizes information from PDF documents using embeddings and a Retrieval-Augmented Generation (RAG) approach.  

---

## ✨ About

This project allows users to search across multiple PDF documents and get the most relevant information in seconds. It leverages **vector embeddings** and a **ChromaDB vector store** to enable fast and accurate retrieval.  

---

## ⚙️ Backend Features

The backend implements a complete RAG pipeline:

- 🗂 **PDF Loading:** Automatically loads all PDFs from a directory using `PyMuPDFLoader`.  
- ✂️ **Text Chunking:** Splits PDF pages into smaller, overlapping chunks for better retrieval.  
- 🤖 **Embedding Generation:** Converts text chunks into vector embeddings using `SentenceTransformer`.  
- 💾 **Vector Store:** Stores embeddings in a **ChromaDB** vector database for fast similarity searches.  
- 🔍 **RAG Retriever:** Retrieves the most relevant chunks for user queries with similarity scoring.  
- 🧮 **Query Embedding:** Converts queries into embeddings and finds top-k relevant chunks.  
- 🛡 **Filtering:** Applies similarity thresholds and deduplication for cleaner results.  
- 🖥 **Frontend Integration:** Works with Streamlit to display search results interactively.  

---

## 🚀 Features

- Search across multiple PDF documents instantly  
- Retrieve top relevant sections with similarity scores  
- Easily add new PDFs to expand your knowledge base  
- Frontend interface powered by **Streamlit** for interactive search  

---

## 💻 How to Use

1. Clone the repository:  
git clone https://github.com/Ilhamlafeer/PDF_RAG-Summarizer.git

2. Install dependencies:
pip install -r requirements.txt

3. Place your PDF files in data/pdf/ directory.

4. Run the Streamlit app:
streamlit run streamlit_app.py

5. Enter your query and see results with summaries and similarity scores.

---

## 🛠 Tech Stack

Python 3.13

Streamlit

PyMuPDF

LangChain

ChromaDB

faiss-cpu

SentenceTransformer


