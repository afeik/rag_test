# update_rag.py
import os
import json
import numpy as np
import faiss
import anthropic
from sentence_transformers import SentenceTransformer
import streamlit as st 

from components.preprocessing import (
    PDFProcessingAgent,
    PaperSummariesIndexAgent,
    EmbeddingIndexAgent,
    spacy_semantic_splitter,
)
from components.llm_agent import ClaudeAgent

# ------------------ CONFIG ------------------ #
PDF_FOLDER = "knowledge_base_solar"
INDEX_FILE = "faiss_index.bin"
TEXT_CHUNKS_FILE = "text_chunks.npy"
SUMMARY_INDEX_FILE = "faiss_index_summaries.bin"
METADATA_FILE = "metadata.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

# If you want summarization
CLAUDE_API_KEY = st.secrets["claude"]["claude_auth"] # or use a secrets manager
client = anthropic.Client(api_key=CLAUDE_API_KEY)
claude_agent = ClaudeAgent(client)

def main():
    print("=== Building & Updating RAG Indexes ===")

    # 1) Process PDFs
    pdf_agent = PDFProcessingAgent(
        folder=PDF_FOLDER,
        splitter_func=spacy_semantic_splitter,
        metadata_file=METADATA_FILE,
    )
    new_doc_chunks, changed = pdf_agent.process_pdfs()
    print(f"PDF Processing complete. Changed: {changed}, new chunks: {len(new_doc_chunks)}")

    # 2) Build or update the chunk-level index
    embedding_agent = EmbeddingIndexAgent(
        model=EMBEDDING_MODEL,
        index_file=INDEX_FILE,
        chunks_file=TEXT_CHUNKS_FILE
    )
    faiss_index, text_chunks = embedding_agent.load_or_create_index()
    if changed and new_doc_chunks:
        faiss_index, text_chunks = embedding_agent.update_index(
            faiss_index, text_chunks, new_doc_chunks
        )
    print(f"Chunk-level index updated. Total chunks: {len(text_chunks)}")

    # 3) Summarize new papers if missing
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    for pdf_file, info in metadata.items():
        # Skip special keys
        if pdf_file == "_paper_id_order":
            continue
        # Summarize if no summary yet
        if isinstance(info, dict) and "summary" not in info and "hash" in info:
            print(f"Summarizing paper: {pdf_file}")
            # Collect only chunks for this PDF
            pdf_chunks = [c["chunk"] for c in new_doc_chunks if c["title"] == pdf_file]
            combined_text = "\n".join(pdf_chunks)[:12000]  # limit length
            summary = claude_agent.summarize_text(combined_text)
            metadata[pdf_file]["summary"] = summary

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    # 4) Build or update paper-level summaries index
    summaries_agent = PaperSummariesIndexAgent(
        model=EMBEDDING_MODEL,
        summary_index_file=SUMMARY_INDEX_FILE,
        metadata_file=METADATA_FILE,
    )
    faiss_index_summaries = summaries_agent.load_or_create_summary_index()
    if faiss_index_summaries:
        summaries_agent.add_new_summaries_to_index()
    print("Paper-level summary index updated.")

    print("=== RAG Update Complete ===")

if __name__ == "__main__":
    main()
