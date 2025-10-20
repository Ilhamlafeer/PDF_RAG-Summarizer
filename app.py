# streamlit_app_pdf_summary.py
import streamlit as st
import sys
import os
import requests
from collections import defaultdict

# Add notebook folder to path
sys.path.append(r"E:\RAG\notebook")
from pdf_loader import rag_retriever

st.set_page_config(page_title="PDF RAG Search", layout="wide")
st.title("📄 PDF RAG Search")

# --- Hugging Face Summarization setup ---
HF_TOKEN = os.getenv("HF_TOKEN")  # Make sure your HF token has Inference API access
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
HF_MODEL_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"

def summarize_text(text: str, max_length: int = 150) -> str:
    """Summarize text using Hugging Face DistilBART model via API"""
    text = text[:1000]  # Keep first 1000 chars per chunk to avoid API issues
    payload = {
        "inputs": text,
        "parameters": {"max_length": max_length, "do_sample": False}
    }
    try:
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=30)
        result = response.json()
        if isinstance(result, list) and "summary_text" in result[0]:
            return result[0]["summary_text"]
        elif isinstance(result, dict) and "error" in result:
            return f"API Error: {result['error']}"
        else:
            return "Could not generate summary."
    except Exception as e:
        return f"Exception: {e}"

# --- User input ---
query = st.text_input("Ask a question:")

if st.button("Search") and query:
    st.session_state['query'] = query
    # Retrieve top 10 chunks to capture best results
    st.session_state['results'] = rag_retriever.retrieve(query, top_k=10)

# --- Display results ---
if 'results' in st.session_state:
    results = st.session_state['results']
    query_display = st.session_state.get('query', '')

    st.subheader(f"Results for: {query_display}")

    if results:
        # Group chunks by PDF
        pdf_chunks = defaultdict(list)
        for res in results:
            pdf_chunks[res['metadata']['source_file']].append(res)

        # Summarize per PDF
        for i, (pdf_name, chunks) in enumerate(pdf_chunks.items(), 1):
            st.markdown(f"### PDF {i}: {pdf_name}")

            # Select chunk with highest similarity for display
            best_chunk = max(chunks, key=lambda x: x['similarity_score'])
            st.markdown(f"- **Top Chunk Similarity:** {best_chunk['similarity_score']:.2f}")
            with st.expander("Show top chunk content"):
                st.write(best_chunk['document'][:500] + "...")

            # Merge all chunk texts into one string for PDF-level summary
            combined_text = " ".join([c['document'] for c in chunks])
            summary = summarize_text(combined_text)
            st.markdown(f"**PDF-level Summary:** {summary}")

            st.markdown("---")
    else:
        st.write("No results found.")
