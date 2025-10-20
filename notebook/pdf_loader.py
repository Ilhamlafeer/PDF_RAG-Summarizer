#!/usr/bin/env python
# coding: utf-8

# ### RAG Pipeline - Data Ingestion to Vector DB Pipeline

# In[2]:


from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


# In[3]:


### Read all the pdf files from the directory
def process_all_pdfs(pdf_directory):
    """Process all PDF files in the given directory."""
    all_documents = []
    pdf_dir = Path(pdf_directory)

    # Find all PDF files in the directory
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process.")

    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file.name}")

        try:
            # Load the PDF file using PyMuPDFLoader
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            # Add source information to metadata
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = 'pdf'

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents

# Process all PDFs in the data directory
all_loaded_documents = process_all_pdfs("E:/RAG/data/pdf")



# In[4]:


### Text Splitting get into chunks
def split_text_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for better RAG performance.

    Parameters:
    - documents: list of Document objects
    - chunk_size: maximum number of characters per chunk
    - chunk_overlap: number of overlapping characters between chunks

    Returns:
    - split_docs: list of chunked Document objects
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Show a sample chunk
    if split_docs:
        print("\nExample chunk")
        print(f"Content: {split_docs[0].page_content[:200]}...")  # first 200 characters
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs


# In[5]:


chunks = split_text_into_chunks(all_loaded_documents)
chunks


# ### Embedding and VectorStoreDB

# In[6]:


import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


# In[8]:


class EmbeddingManager:
    """Handles document embeddings using SentenceTransformer"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the Embedding Manager

        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Embedding model is not loaded.")

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

## Initialize the embedding manager
embedding_manager = EmbeddingManager()


# ### Vector Store

# In[11]:


import os
class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "E:/RAG/data/vector_store"):
        """
        Initialize the Vector Store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the ChromaDB data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize the ChromaDB client and collection"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF Document Embeddings for RAG"}
            )
            print(f"Vector store initialized with collection: {self.collection_name}")
            print(f"Existing documents in store: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.

        Args:
            documents: List of LangChain Documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents and embeddings must match.")

        print(f"Adding {len(documents)} documents to the vector store...")

        ids = []
        metadatas = []
        document_text = []
        embedding_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            document_text.append(doc.page_content)
            embedding_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=document_text,
                embeddings=embedding_list
            )
            print(f"Successfully added {len(documents)} documents to the vector store")
            print(f"Total documents in store: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

# Initialize vector store
vectorstore = VectorStore()
vectorstore


# In[12]:


chunks


# In[13]:


### Convert the text to embeddings
texts = [doc.page_content for doc in chunks]

# Genrerate embeddings
embeddings = embedding_manager.generate_embeddings(texts)

# Store in vector DB
vectorstore.add_documents(chunks, embeddings)


# ### Retriever Pipeline From VectorStore

# In[14]:


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the RAG Retriever

        Args:
            vector_store: VectorStore containing document embeddings
            embedding_manager: Manager for generating query embeddings
            top_k: Number of top similar documents to retrieve
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for the given query

        Args:
            query: The search query
            top_k: Number of top similar documents to retrieve
            score_threshold: Minimum similarity score to consider

        Returns:
            List of dictionaries containing retrieved documents and their metadata
        """

        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score Threshold: {score_threshold}")

        # Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in the vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found.")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

rag_retriever = RAGRetriever(vectorstore, embedding_manager)


# In[15]:


rag_retriever


# In[16]:


rag_retriever.retrieve("What is Machine Learning?")


# ### Integration VectorDB Context pipeline with LLM Output

# In[17]:


### Simple RAG pipeline with Groq LLM
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

### Initialize Groq LLM (API Key from environment)
groq_api_key = os.getenv("groq_api_key")

llm = ChatGroq(groq_api_key=groq_api_key, model_name = "llama-3.1-8b-instant",temperature=0.1, max_tokens=1024)

### Simple RAG Function: retrieve context and generate answer
def rag_simple(query, retriever, llm, top_k=3):
    ## Retrieve the context
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return "No relevant context found to answer the query."

    context = "\n\n".join(
        [doc.page_content if hasattr(doc, "page_content") else doc.get("content", "") for doc in results]
    )

    if not context:
        return "No relevant context found to answer the query."

    ## Generate the answer using groq LLM
    prompt = f"""Use the following context to answer the question concisely.
        context:
        {context}

        question: {query}
        Answer:"""

    response = llm.invoke(prompt)
    return response.content


# In[18]:


answer = rag_simple("What is supervised learning", rag_retriever, llm)
print(answer)


# ### Enhanced RAG Pipeline features

# In[19]:


def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {"answer": "No relevant context found.", "sources": [], "confidence": 0.0, "context": ""}

    # Safely extract context and sources
    context_list = []
    sources = []
    for doc in results:
        # Get text
        if isinstance(doc, dict):
            text = doc.get('content') or doc.get('page_content') or doc.get('text') or ""
            metadata = doc.get('metadata', {})
            score = doc.get('similarity_score', 1.0)
        else:  # Document object
            text = getattr(doc, 'page_content', "")
            metadata = getattr(doc, 'metadata', {})
            score = getattr(doc, 'similarity_score', 1.0)

        context_list.append(text)
        sources.append({
            "source": metadata.get("source_file", metadata.get("source", "unknown")),
            "page": metadata.get("page", "unknown"),
            "preview": text[:120] + "...",
            "score": score
        })

    context = "\n\n".join(context_list)
    confidence = max(s['score'] for s in sources) if sources else 0.0

    # Generate answer
    prompt = f"""Use the following context to answer the question concisely.

Context:
{context}

Question: {query}

Answer:
"""
    response = llm.invoke(prompt)

    output = {"answer": response.content, "sources": sources, "confidence": confidence}
    if return_context:
        output["context"] = context

    return output

# Example usage
result = rag_advanced("Explain machine learning", rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)
print("Answer:", result["answer"])
print("Sources:", result["sources"])
print("Confidence:", result["confidence"])
print("Context Preview:", result["context"][:300])

