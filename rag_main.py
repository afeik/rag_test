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

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

CLAUDE_API_KEY = st.secrets.get("claude", {}).get("claude_auth", "YOUR_CLAUDE_API_KEY")
client = anthropic.Client(api_key=CLAUDE_API_KEY)

PDF_FOLDER = "knowledge_base"        
INDEX_FILE = "faiss_index.bin"       
TEXT_CHUNKS_FILE = "text_chunks.npy" 
METADATA_FILE = "metadata.json"      

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Paper titles
titles = [
    "Swiss Energy System 2050: Pathways to Net Zero CO 2 and Security of Supply - A basic report, Swiss Academies of Arts and Sciences",
    "Policy-relevance of a Model Inter-comparison: Switzerland in the European Energy Transition - Sansavini et al.",
    "Energy security in a net zero emissions future for Switzerland Expert Group (Security of Supply) – White Paper - Hug, Patt et al.",
    "High resolution generation expansion planning considering flexibility needs: The case of Switzerland in 2030 - G. Hug et al.",
    "The Swiss energy transition: Policies to address the Energy Trilemma, Georges, Gil; Boulouchos, Konstantinos et al.",
    "The role of digital social practices and technologies in the Swiss energy transition towards net-zero carbon dioxide emissions in 2050 - Panos et al.",
    "A historical turning point? Early evidence on how the Russia-Ukraine war changes public support for clean energy policies - Patt, Anthony et al.",
    "Social and environmental policy in sustainable energy transition - Yulia Ermolaeva",
    "Phases of the net-zero energy transition and strategies to achieve it - Jochen Markard and Daniel Rosenbloom",
    "Geopolitical dimensions of the energy transition - Kamasa Julian",
    "Navigating the clean energy transition in the COVID-19 crisis - Bjarne Steffen, Tobias M Schmidt et al.",
    "Of renewable energy, energy democracy, and sustainable development: A roadmap to accelerate the energy transition in developing countries - Cantarero et al.",
    "Energy requirements and carbon emissions for a low-carbon energy transition - Nature Communications - Daniel W. O’Neill et al."
]

########################################
#         SEMANTIC SPLITTING           #
########################################

def spacy_semantic_splitter(text, max_chunk_size=1500, chunk_overlap=50):
    """
    Splits the text semantically using spaCy.
    """
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

    return chunks

########################################
#         AGENTS / CLASSES             #
########################################

class PDFProcessingAgent:
    def __init__(self, folder, splitter_func, metadata_file):
        self.folder = folder
        self.splitter_func = splitter_func
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_file_hash(self, filepath):
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
                # Normalize text encoding
                full_text += page_text.encode("utf-8", errors="replace").decode("utf-8")
        return full_text

    def _extract_from_abstract(self, full_text):
        """
        Attempt to extract content starting from the abstract and excluding references.
        """
        abstract_start = re.search(r"(?i)\babstract\b", full_text)
        references_start = re.search(r"(?i)\breferences\b", full_text)

        if abstract_start:
            start_idx = abstract_start.start()
            end_idx = references_start.start() if references_start else len(full_text)
            return full_text[start_idx:end_idx]
        else:
            # Fallback: Process the entire document
            return full_text

    def process_pdfs(self):
        text_chunks = []
        changed = False

        pdf_files = [f for f in os.listdir(self.folder) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.folder, pdf_file)
            file_hash = self._compute_file_hash(pdf_path)

            if pdf_file not in self.metadata or self.metadata[pdf_file] != file_hash:
                raw_text = self._extract_text_from_pdf(pdf_path)
                processed_text = self._extract_from_abstract(raw_text)
                chunks = self.splitter_func(processed_text)
                text_chunks.extend(chunks)
                self.metadata[pdf_file] = file_hash
                changed = True

        # Remove metadata for deleted files
        existing_files = set(pdf_files)
        for file in list(self.metadata.keys()):
            if file not in existing_files:
                del self.metadata[file]
                changed = True

        self._save_metadata()
        return text_chunks, changed

class EmbeddingIndexAgent:
    def __init__(self, model, index_file, chunks_file):
        self.model = model
        self.index_file = index_file
        self.chunks_file = chunks_file

    def load_or_create_index(self):
        if os.path.exists(self.index_file):
            return faiss.read_index(self.index_file), np.load(self.chunks_file, allow_pickle=True).tolist()
        return None, []

    def update_index(self, faiss_index, text_chunks, new_text_chunks):
        new_embeddings = [self.model.encode(chunk) for chunk in new_text_chunks]

        if faiss_index is None:
            dimension = len(new_embeddings[0])
            faiss_index = faiss.IndexFlatIP(dimension)
            faiss_index.add(np.array(new_embeddings))
        else:
            faiss_index.add(np.array(new_embeddings))

        faiss.write_index(faiss_index, self.index_file)
        np.save(self.chunks_file, text_chunks + new_text_chunks)

        return faiss_index, text_chunks + new_text_chunks

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
    st.title("Energy Transition Knowledge Base")

    st.sidebar.header("Overview of Papers")
    for title in titles:
        st.sidebar.write(f"- {title}")

    st.sidebar.write("\n---\n**Developed for insights into the energy transition.")

    splitter_func = spacy_semantic_splitter
    pdf_agent = PDFProcessingAgent(PDF_FOLDER, splitter_func, METADATA_FILE)
    embedding_agent = EmbeddingIndexAgent(EMBEDDING_MODEL, INDEX_FILE, TEXT_CHUNKS_FILE)

    with st.spinner("Loading or creating the FAISS Database index..."):
        faiss_index, text_chunks = embedding_agent.load_or_create_index()

    with st.spinner("Processing PDFs..."):
        new_text_chunks, changed = pdf_agent.process_pdfs()

    if changed:
        with st.spinner("Updating the FAISS index..."):
            faiss_index, text_chunks = embedding_agent.update_index(faiss_index, text_chunks, new_text_chunks)

    st.header("Ask a Question")
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Searching for relevant information and generating response..."):
            query_embeddings = EMBEDDING_MODEL.encode([query])
            distances, indices = faiss_index.search(query_embeddings, k=5)

            # Deduplicate chunks
            seen_chunks = set()
            results = []
            for idx in indices[0]:
                chunk = text_chunks[idx]
                chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
                if chunk_hash not in seen_chunks:
                    seen_chunks.add(chunk_hash)
                    results.append(chunk)

            claude_agent = ClaudeAgent(client)
            context_str = "\n\n".join(results)
            response = claude_agent.call_claude(query, context_str)

        with st.expander("View Retrieved Chunks"):
            for i, result in enumerate(results):
                st.write(f"**Chunk {i+1}:** {result}")

        st.subheader("Claude's Response")
        st.write(response)

if __name__ == "__main__":
    main()
