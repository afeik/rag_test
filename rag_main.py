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

# Attempt to load spaCy model or download if not present
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

# Paper titles (manually stored). 
# This list is mostly for display in the sidebar.
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
    Splits by sentence boundary but keeps chunk sizes around 1,500 characters.
    Overlapping tokens can be introduced if desired.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence exceeds the max_chunk_size, close out the current chunk
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            # Save current chunk
            chunks.append(current_chunk.strip())
            # Optionally add overlap if needed 
            # For now, we skip the overlap
            current_chunk = sentence

    # Add the last chunk if any
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

    def _extract_title_from_filename(self, pdf_file):
        """
        Parse the filename to get the title (and possibly authors).
        Example filename: 'Title_of_the_paper_authorX_authorY.pdf'
        """
        base_name = os.path.splitext(pdf_file)[0]  # remove .pdf
        # Replace underscores with spaces
        pdf_title = base_name.replace("_", " ")
        # Optional:
        # If you want to separate out authors, you can split by underscore:
        # parts = base_name.split("_")
        # title_part = " ".join(parts[:-2])  # e.g., everything except last 1–2 tokens
        # authors_part = ", ".join(parts[-2:])
        # pdf_title = f"{title_part} (by {authors_part})"
        return pdf_title.strip()

    def _extract_text_from_pdf(self, pdf_path):
        """
        Extract raw text from PDF, removing typical noise like 
        page headers, footers, etc.
        """
        reader = PdfReader(pdf_path)
        text_chunks = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Remove typical footers/headers like "Page x of y"
                page_text = re.sub(r"Page\s*\d+(\s+of\s+\d+)?", "", page_text, flags=re.IGNORECASE)
                # Remove purely numeric lines (common for PDF page numbers)
                page_text = re.sub(r"^\s*\d+\s*$", "", page_text, flags=re.MULTILINE)
                page_text = page_text.encode("utf-8", errors="replace").decode("utf-8")
                text_chunks.append(page_text)

        full_text = "\n".join(text_chunks)
        return reader, full_text

    def _extract_from_abstract(self, full_text):
        """
        Attempt to filter text starting from 'Abstract', excluding 
        'References' (optional). 
        Remove disclaimers, repeated newlines, etc.
        """
        # Remove disclaimers like '© 20XX ...'
        full_text = re.sub(r"©\s*20\d{2}.*", "", full_text, flags=re.IGNORECASE)

        # Find 'abstract' and 'references' ignoring case
        abstract_pattern = r"(?i)\babstract\b"
        references_pattern = r"(?i)\breferences\b"

        lowered_text = full_text.lower()
        abstract_match = re.search(abstract_pattern, lowered_text)
        references_match = re.search(references_pattern, lowered_text)

        if abstract_match:
            start_idx = abstract_match.start()
        else:
            start_idx = 0

        if references_match:
            end_idx = references_match.start()
        else:
            end_idx = len(full_text)

        extracted_text = full_text[start_idx:end_idx]

        # Clean up leftover newline duplications or extra spacing
        extracted_text = re.sub(r"\n\s*\n+", "\n\n", extracted_text)
        return extracted_text.strip()

    def process_pdfs(self):
        """
        Process each PDF in the folder:
        1. Check if new/changed by comparing file hash.
        2. Extract text from Abstract onward.
        3. Split into semantic chunks.
        4. Save {chunk, title} to text_chunks.
        5. Update local metadata.
        """
        text_chunks = []
        changed = False

        pdf_files = [f for f in os.listdir(self.folder) if f.lower().endswith(".pdf")]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.folder, pdf_file)
            file_hash = self._compute_file_hash(pdf_path)

            # If it's a new or changed file
            if pdf_file not in self.metadata or self.metadata[pdf_file].get("hash") != file_hash:
                # In this approach, we always get the PDF title from the filename
                pdf_title = self._extract_title_from_filename(pdf_file)

                # Extract the full text
                reader, raw_text = self._extract_text_from_pdf(pdf_path)

                # Filter from abstract to references if possible
                processed_text = self._extract_from_abstract(raw_text)

                # Now split the processed text into smaller semantic chunks
                chunks = self.splitter_func(processed_text)

                # Store each chunk with its title
                doc_chunks = [{"chunk": chunk, "title": pdf_title} for chunk in chunks]
                text_chunks.extend(doc_chunks)

                # Update metadata
                self.metadata[pdf_file] = {
                    "hash": file_hash,
                    "title": pdf_title
                }
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
        """
        Attempt to load an existing FAISS index and text chunks.
        If it doesn't exist, return None, [].
        """
        if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
            faiss_index = faiss.read_index(self.index_file)
            text_chunks = np.load(self.chunks_file, allow_pickle=True).tolist()
            return faiss_index, text_chunks
        return None, []

    def update_index(self, faiss_index, existing_text_chunks, new_doc_chunks):
        """
        - Convert the new chunks to embeddings
        - Add them to the FAISS index
        - Save to disk
        Returns the updated FAISS index and updated text_chunks
        """
        # Prepare only the chunk texts for embeddings
        new_texts = [c["chunk"] for c in new_doc_chunks]
        new_embeddings = self.model.encode(new_texts, show_progress_bar=False)

        # If no existing index, create one from scratch using an Approximate Nearest Neighbor factory
        if faiss_index is None:
            dimension = len(new_embeddings[0])
            # Example: HNSW index for faster approximate search
            # (You can experiment with other factories like "IVF100,PQ64" or GPU-based indices.)
            faiss_index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW with 32 neighbors
            faiss_index.hnsw.efConstruction = 80
            faiss_index.hnsw.efSearch = 80
            faiss_index.metric_type = faiss.METRIC_INNER_PRODUCT

            # We need to normalize embeddings if we plan to use METRIC_INNER_PRODUCT effectively
            # This is optional but often done with sentence-transformers + IP metric
            new_embeddings = np.array([x / np.linalg.norm(x) for x in new_embeddings], dtype="float32")
            faiss_index.add(new_embeddings)
        else:
            # If reusing an IP index, remember to normalize the new vectors
            new_embeddings = np.array([x / np.linalg.norm(x) for x in new_embeddings], dtype="float32")
            faiss_index.add(new_embeddings)

        # Save updated index and combined chunks
        updated_text_chunks = existing_text_chunks + new_doc_chunks
        faiss.write_index(faiss_index, self.index_file)
        np.save(self.chunks_file, updated_text_chunks)

        return faiss_index, updated_text_chunks

class ClaudeAgent:
    def __init__(self, client):
        self.client = client

    def call_claude(self, query, context):
        """
        Format a prompt for Claude and send a request.
        """
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
        
        # Return the textual content from Claude
        return response.content[0].text.strip()

########################################
#              STREAMLIT APP           #
########################################

def main():
    st.title("Energy Transition Knowledge Base")

    # Sidebar titles overview
    st.sidebar.header("Overview of Papers")
    for title in titles:
        st.sidebar.write(f"- {title}")

    st.sidebar.write("\n---\n**Developed for insights into the energy transition.**")

    splitter_func = spacy_semantic_splitter
    pdf_agent = PDFProcessingAgent(PDF_FOLDER, splitter_func, METADATA_FILE)
    embedding_agent = EmbeddingIndexAgent(EMBEDDING_MODEL, INDEX_FILE, TEXT_CHUNKS_FILE)

    # Check for the presence of the PDF folder
    if not os.path.exists(PDF_FOLDER):
        #st.warning(f"PDF folder '{PDF_FOLDER}' not found. Upload or create it for processing.")
        pdf_files_present = False
    else:
        pdf_files_present = True

    # Load or create FAISS index
    with st.spinner("Loading or creating the FAISS Database index..."):
        try:
            faiss_index, text_chunks = embedding_agent.load_or_create_index()
        except Exception as e:
            st.error(f"Error loading FAISS index or text chunks: {e}")
            faiss_index = None
            text_chunks = []

    # Process PDFs if folder exists
    if pdf_files_present:
        with st.spinner("Processing PDFs..."):
            try:
                new_doc_chunks, changed = pdf_agent.process_pdfs()
                if changed and new_doc_chunks:
                    with st.spinner("Updating the FAISS index..."):
                        faiss_index, text_chunks = embedding_agent.update_index(faiss_index, text_chunks, new_doc_chunks)
            except Exception as e:
                st.error(f"Error processing PDFs: {e}")

    # Ask a question
    st.header("Ask a Question")
    query = st.text_input("Enter your question:")

    if query:
        if faiss_index is None:
            st.error("No FAISS index is available. Ensure the PDF files are uploaded and processed.")
        else:
            with st.spinner("Searching for relevant information and generating response..."):
                try:
                    # Convert query to embedding
                    query_embedding = EMBEDDING_MODEL.encode([query], show_progress_bar=False)
                    query_embedding = np.array([x / np.linalg.norm(x) for x in query_embedding], dtype="float32")

                    # Retrieve top-5 relevant chunks 
                    distances, indices = faiss_index.search(query_embedding, k=5)

                    # Deduplicate chunks to avoid repeated identical text
                    seen_hashes = set()
                    retrieved_results = []
                    for idx in indices[0]:
                        doc_chunk = text_chunks[idx]
                        chunk_text = doc_chunk["chunk"]
                        chunk_title = doc_chunk.get("title", "Unknown Title")
                        chunk_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
                        if chunk_hash not in seen_hashes:
                            seen_hashes.add(chunk_hash)
                            retrieved_results.append(doc_chunk)

                    # Build a single context string for Claude
                    # Optionally, we could combine the top n chunks with some formatting.
                    context_parts = []
                    for i, res in enumerate(retrieved_results, start=1):
                        part_title = res.get("title", "Unknown Title")
                        part_text = res["chunk"]
                        context_parts.append(f"Chunk {i} (from {part_title}):\n{part_text}")
                    context_str = "\n\n".join(context_parts)

                    # Query Claude
                    claude_agent = ClaudeAgent(client)
                    response = claude_agent.call_claude(query, context_str)

                    # Display retrieval
                    with st.expander("View Retrieved Chunks"):
                        for i, res in enumerate(retrieved_results, start=1):
                            st.write(f"**Chunk {i} (from {res.get('title', 'Unknown Title')}):**\n{res['chunk']}")

                    st.subheader("Claude's Response")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error during query processing: {e}")


if __name__ == "__main__":
    main()
