import os
import hashlib
import json
import numpy as np
import streamlit as st
import faiss
import anthropic
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import spacy
import re

########################################
#            CONFIG & SETUP            #
########################################

CLAUDE_API_KEY = st.secrets.get("claude", {}).get("claude_auth", "YOUR_CLAUDE_API_KEY")
client = anthropic.Client(api_key=CLAUDE_API_KEY)

PDF_FOLDER = "knowledge_base"        
INDEX_FILE = "faiss_index.bin"       
TEXT_CHUNKS_FILE = "text_chunks.npy" 
METADATA_FILE = "metadata.json"      

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load spaCy model for semantic splitting
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("spaCy model not found. Installing 'en_core_web_sm'...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

########################################
#         SEMANTIC SPLITTING           #
########################################

def spacy_semantic_splitter(
    text, 
    max_chunk_size=1500, 
    chunk_overlap=50,  # Smaller overlap to reduce embedding count
    ignore_references=True, 
    remove_citations=True
):
    """
    Custom splitter for academic papers with preprocessing options.
    """
    if ignore_references and "References" in text:
        text = text.split("References")[0]

    if remove_citations:
        text = re.sub(r"\([A-Za-z]+( et al\.)?,?\s?\d{4}\)", "", text)
        text = re.sub(r"\[\d+\]", "", text)
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    if chunk_overlap > 0:
        new_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                new_chunks.append(chunk)
            else:
                overlap_part = chunks[i-1][-chunk_overlap:]
                new_chunks.append(overlap_part + " " + chunk)
        chunks = [c.strip() for c in new_chunks]

    return [c for c in chunks if c]

########################################
#         AGENTS / CLASSES             #
########################################

class PDFProcessingAgent:
    def __init__(self, folder, splitter_func, metadata_file):
        self.folder = folder
        self.splitter_func = splitter_func
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    @staticmethod
    def _compute_file_hash(filepath):
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _extract_text_from_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text
        return full_text

    def process_pdfs(self):
        text_chunks = []
        changed = False

        if not os.path.exists(self.folder):
            st.warning(f"PDF folder '{self.folder}' does not exist. Skipping PDF processing.")
            return text_chunks, changed

        pdf_files = [f for f in os.listdir(self.folder) if f.endswith(".pdf")]
        st.write(f"Found {len(pdf_files)} PDF(s) in '{self.folder}'.")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.folder, pdf_file)
            file_hash = self._compute_file_hash(pdf_path)

            if (pdf_file not in self.metadata) or (self.metadata[pdf_file] != file_hash):
                st.write(f"Processing new or updated PDF: {pdf_file}")
                raw_text = self._extract_text_from_pdf(pdf_path)
                chunks = self.splitter_func(raw_text)
                self.metadata[pdf_file] = file_hash
                text_chunks.extend(chunks)
                changed = True

        removed_files = [f for f in self.metadata if f not in pdf_files]
        for removed_file in removed_files:
            del self.metadata[removed_file]
            changed = True

        if changed:
            self._save_metadata()

        return text_chunks, changed

class EmbeddingIndexAgent:
    def __init__(self, model, index_file, chunks_file):
        self.model = model
        self.index_file = index_file
        self.chunks_file = chunks_file

    @staticmethod
    def normalize_vectors(vectors):
        return np.array([v / np.linalg.norm(v) for v in vectors])

    def load_or_create_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
            st.write("Loading existing FAISS index & text chunks...")
            faiss_index = faiss.read_index(self.index_file)
            text_chunks = np.load(self.chunks_file, allow_pickle=True).tolist()
            return faiss_index, text_chunks
        else:
            st.write("No existing index found. Creating a new one...")
            return None, []

    def update_index(self, faiss_index, text_chunks, new_text_chunks):
        st.write(f"Generating embeddings for {len(new_text_chunks)} new chunks...")
        new_embeddings = [self.model.encode(chunk) for chunk in new_text_chunks]
        new_embeddings = self.normalize_vectors(np.array(new_embeddings))

        if faiss_index is None:
            dimension = new_embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)
            faiss_index.add(new_embeddings)
            text_chunks = new_text_chunks
        else:
            faiss_index.add(new_embeddings)
            text_chunks.extend(new_text_chunks)

        faiss.write_index(faiss_index, self.index_file)
        np.save(self.chunks_file, text_chunks)

        return faiss_index, text_chunks

class QueryAgent:
    def __init__(self, model, faiss_index, text_chunks):
        self.model = model
        self.faiss_index = faiss_index
        self.text_chunks = text_chunks

    def query(self, user_query, top_k=5):
        query_embedding = self.model.encode(user_query)
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        distances, indices = self.faiss_index.search(
            np.array([normalized_query]), k=top_k
        )
        results = [self.text_chunks[i] for i in indices[0]]
        return results

class ClaudeAgent:
    def __init__(self, client):
        self.client = client

    def call_claude(self, query, context):
        prompt = (
            f"{anthropic.HUMAN_PROMPT}"
            f"Use the following context to answer the question:\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"{anthropic.AI_PROMPT}"
        )

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()

########################################
#              STREAMLIT APP           #
########################################

def main():
    st.title("RAG System for some energy transition papers")
    st.write("Processes PDFs and answers queries with Claude.")

    splitter_func = spacy_semantic_splitter
    pdf_agent = PDFProcessingAgent(PDF_FOLDER, splitter_func, METADATA_FILE)
    embedding_agent = EmbeddingIndexAgent(EMBEDDING_MODEL, INDEX_FILE, TEXT_CHUNKS_FILE)

    faiss_index, text_chunks = embedding_agent.load_or_create_index()

    with st.spinner("Processing PDFs..."):
        new_text_chunks, changed = pdf_agent.process_pdfs()

    if changed and new_text_chunks:
        with st.spinner("Updating the FAISS index..."):
            faiss_index, text_chunks = embedding_agent.update_index(
                faiss_index=faiss_index,
                text_chunks=text_chunks,
                new_text_chunks=new_text_chunks
            )

    st.write("## Ask a Question")
    query = st.text_input("Enter your query here:")
    if query:
        if faiss_index is None or not text_chunks:
            st.warning("No FAISS index created yet or no chunks available.")
        else:
            query_agent = QueryAgent(EMBEDDING_MODEL, faiss_index, text_chunks)
            retrieved_chunks = query_agent.query(query, top_k=5)

            st.write("### Retrieved Context")
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.write(f"**Chunk {i}:** {chunk}")

            with st.spinner("Generating response with Claude..."):
                claude_agent = ClaudeAgent(client)
                context_str = "\n\n".join(retrieved_chunks)
                response = claude_agent.call_claude(query, context_str)
                st.write("### Claude's Response")
                st.write(response)

if __name__ == "__main__":
    main()
