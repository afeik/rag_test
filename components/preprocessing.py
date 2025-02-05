# components/preprocessing.py

import os
import re
import json
import hashlib
import numpy as np
import spacy
import nltk
import faiss
import fitz  # PyMuPDF
from nltk.corpus import wordnet as wn
from nltk import download as nltk_download
from nltk.data import find
from sentence_transformers import SentenceTransformer

########################################
#            CONFIG & SETUP            #
########################################

def load_or_download_spacy(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"spaCy model '{model_name}' not found. Downloading...")
        os.system(f"python -m spacy download {model_name}")
        return spacy.load(model_name)

nlp = load_or_download_spacy()

def ensure_nltk_wordnet():
    nltk_resources = ["wordnet", "omw-1.4"]
    for resource in nltk_resources:
        try:
            find(f"corpora/{resource}")
        except LookupError:
            print(f"NLTK resource '{resource}' not found. Downloading...")
            nltk_download(resource)

ensure_nltk_wordnet()

########################################
#         SEMANTIC SPLITTING           #
########################################

def spacy_semantic_splitter(text, max_chunk_size=1500, chunk_overlap=50):
    """
    Splits text into semantic chunks using spaCy sentence boundaries,
    with optional overlap in characters to retain context between chunks.

    Args:
        text (str): The full text to chunk.
        max_chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Number of trailing characters from the previous
                             chunk to carry over as context.

    Returns:
        list: A list of chunk strings.
    """

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            chunks.append(current_chunk.strip())
            # Overlap logic
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                overlap_text = current_chunk[-chunk_overlap:]
            else:
                overlap_text = ""
            current_chunk = overlap_text + " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

########################################
#         QUERY PREPROCESSING          #
########################################

def preprocess_user_query(query, custom_synonyms=None, max_synonyms=2):
    """
    Normalizes and expands the user's query with synonyms from custom dictionary and WordNet.
    """
    query = re.sub(r"[^a-zA-Z0-9\s]", " ", query)
    doc = nlp(query)

    processed_query = []
    for token in doc:
        if token.is_stop:
            processed_query.append(token.text)
            continue

        original_lower = token.text.lower()
        lemma_lower = token.lemma_.lower()

        synonyms = {token.text}

        # Custom synonyms if provided
        if custom_synonyms:
            if original_lower in custom_synonyms:
                synonyms.update(custom_synonyms[original_lower])
            if lemma_lower in custom_synonyms:
                synonyms.update(custom_synonyms[lemma_lower])

        # WordNet synonyms
        if token.pos_ in {"NOUN", "VERB", "ADJ"} and len(synonyms) <= max_synonyms:
            pos = (
                wn.NOUN if token.pos_ == "NOUN"
                else wn.VERB if token.pos_ == "VERB"
                else wn.ADJ
            )
            for synset in wn.synsets(lemma_lower, pos=pos):
                for syn in synset.lemma_names():
                    syn = syn.replace("_", " ")
                    if syn.lower() not in [s.lower() for s in synonyms]:
                        synonyms.add(syn)
                    if len(synonyms) >= (max_synonyms + 1):
                        break
                if len(synonyms) >= (max_synonyms + 1):
                    break

        synonyms_list = sorted(synonyms, key=lambda s: s.lower())
        synonyms_joined = " / ".join(synonyms_list[: (max_synonyms + 1)])
        processed_query.append(synonyms_joined)

    return " ".join(processed_query)

########################################
#        PDF PROCESSING AGENT          #
########################################

class PDFProcessingAgent:
    def __init__(self, folder, splitter_func, metadata_file):
        """
        Args:
            folder (str): Path to the folder containing PDFs
            splitter_func (callable): A function to split text into chunks
            metadata_file (str): JSON file to track PDF metadata
        """
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

    def _extract_text_pymupdf(self, pdf_path):
        """
        Use PyMuPDF (fitz) to extract text from PDF pages.
        """
        try:
            doc = fitz.open(pdf_path)
        except AttributeError:
            doc = fitz.Document(pdf_path)

        all_text = []
        for page in doc:
            page_text = page.get_text("text") or ""
            page_text = re.sub(r"Page\s*\d+(\s+of\s+\d+)?", "", page_text, flags=re.IGNORECASE)
            page_text = re.sub(r"^\s*\d+\s*$", "", page_text, flags=re.MULTILINE)
            page_text = page_text.encode("utf-8", errors="replace").decode("utf-8")
            if page_text.strip():
                all_text.append(page_text)

        return "\n".join(all_text)

    def _extract_from_abstract(self, full_text):
        """
        Extracts text from 'Abstract' to 'References'.
        If 'Abstract' is not found, extracts from the beginning to 'References'.
        If 'References' is not found, extracts from 'Abstract' to the end.
        If neither is found, returns the full text.
        """
        # Remove copyright notice
        full_text = re.sub(r"©\s*20\d{2}.*", "", full_text, flags=re.IGNORECASE)

        abstract_pattern = r"(?i)\babstract\b"
        references_pattern = r"(?i)\breferences\b"

        abstract_match = re.search(abstract_pattern, full_text)
        references_match = re.search(references_pattern, full_text)

        if abstract_match and references_match:
            start_idx = abstract_match.end()
            end_idx = references_match.start()
            extracted_text = full_text[start_idx:end_idx]
        elif abstract_match:
            start_idx = abstract_match.end()
            extracted_text = full_text[start_idx:]
        elif references_match:
            end_idx = references_match.start()
            extracted_text = full_text[:end_idx]
        else:
            extracted_text = full_text

        # Clean up multiple newlines
        extracted_text = re.sub(r"\n\s*\n+", "\n\n", extracted_text)
        return extracted_text.strip()

    def _extract_title_from_filename(self, filename):
        return os.path.splitext(filename)[0]

    def process_pdfs(self):
        """
        - Scan folder for PDFs
        - For each new or changed PDF, extract text and split into chunks
        - Return newly added chunks
        """
        text_chunks = []
        changed = False

        pdf_files = [f for f in os.listdir(self.folder) if f.lower().endswith(".pdf")]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.folder, pdf_file)
            file_hash = self._compute_file_hash(pdf_path)

            # If new or changed
            if pdf_file not in self.metadata or self.metadata[pdf_file].get("hash") != file_hash:
                raw_text = self._extract_text_pymupdf(pdf_path)
                processed_text = self._extract_from_abstract(raw_text)

                print(f"Processing '{pdf_file}': Extracted text length = {len(processed_text)} characters.")

                chunks = self.splitter_func(processed_text)
                doc_chunks = [{"chunk": chunk, "title": pdf_file} for chunk in chunks]
                text_chunks.extend(doc_chunks)

                if pdf_file not in self.metadata:
                    self.metadata[pdf_file] = {}
                self.metadata[pdf_file]["hash"] = file_hash
                self.metadata[pdf_file]["title"] = pdf_file
                changed = True

        # Clean up metadata for removed PDFs
        existing_files = set(pdf_files)
        for file in list(self.metadata.keys()):
            if file not in existing_files and file != "_paper_id_order":
                del self.metadata[file]
                changed = True

        self._save_metadata()
        return text_chunks, changed

########################################
#   PAPER SUMMARIES & TWO-STAGE INDEX  #
########################################

class PaperSummariesIndexAgent:
    """
    Maintains a separate FAISS index for paper-level summaries.
    """
    def __init__(self, model, summary_index_file, metadata_file):
        self.model = model
        self.summary_index_file = summary_index_file
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()
        self.faiss_index_summaries = None
        self.paper_ids = []

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load_or_create_summary_index(self):
        """
        If summary index file exists, read it. Otherwise build a new index from existing summaries.
        """
        if os.path.exists(self.summary_index_file):
            self.faiss_index_summaries = faiss.read_index(self.summary_index_file)
            self.paper_ids = self.metadata.get("_paper_id_order", [])
        else:
            summaries, self.paper_ids = [], []
            for pdf_file, info in self.metadata.items():
                if isinstance(info, dict) and "summary" in info:
                    summaries.append(info["summary"])
                    self.paper_ids.append(pdf_file)

            if not summaries:
                return None

            summary_embeddings = self.model.encode(summaries, show_progress_bar=False)
            summary_embeddings = np.array(
                [emb / np.linalg.norm(emb) for emb in summary_embeddings],
                dtype="float32"
            )

            dimension = summary_embeddings.shape[1]
            # Using HNSWFlat for summaries
            self.faiss_index_summaries = faiss.IndexHNSWFlat(dimension, 32)
            self.faiss_index_summaries.hnsw.efConstruction = 80
            self.faiss_index_summaries.hnsw.efSearch = 80
            self.faiss_index_summaries.metric_type = faiss.METRIC_INNER_PRODUCT
            self.faiss_index_summaries.add(summary_embeddings)

            faiss.write_index(self.faiss_index_summaries, self.summary_index_file)
            self.metadata["_paper_id_order"] = self.paper_ids
            self._save_metadata()

        return self.faiss_index_summaries

    def update_summary_in_metadata(self, pdf_file, summary):
        if pdf_file not in self.metadata:
            self.metadata[pdf_file] = {}
        self.metadata[pdf_file]["summary"] = summary
        self._save_metadata()

    def add_new_summaries_to_index(self):
        """
        If new summaries exist in metadata but not in the existing index, add them.
        """
        if self.faiss_index_summaries is None:
            return self.load_or_create_summary_index()

        new_summaries = []
        new_ids = []
        for pdf_file, info in self.metadata.items():
            if pdf_file == "_paper_id_order":
                continue
            if pdf_file not in self.paper_ids and "summary" in info:
                new_summaries.append(info["summary"])
                new_ids.append(pdf_file)

        if not new_summaries:
            return self.faiss_index_summaries

        summary_embeddings = self.model.encode(new_summaries, show_progress_bar=False)
        summary_embeddings = np.array(
            [emb / np.linalg.norm(emb) for emb in summary_embeddings],
            dtype="float32"
        )
        self.faiss_index_summaries.add(summary_embeddings)
        self.paper_ids.extend(new_ids)

        faiss.write_index(self.faiss_index_summaries, self.summary_index_file)
        self.metadata["_paper_id_order"] = self.paper_ids
        self._save_metadata()

        return self.faiss_index_summaries

    def search_paper_summaries(self, query_vector, k=3):
        """
        Stage 1 retrieval: find top-k relevant paper summaries.
        Returns a list of PDF filenames in descending order of relevance.
        """
        if self.faiss_index_summaries is None or len(self.paper_ids) == 0:
            return []

        if len(self.paper_ids) != self.faiss_index_summaries.ntotal:
            print("Mismatch in paper_ids vs. index total. Rebuilding summary index.")
            self.faiss_index_summaries = self.load_or_create_summary_index()

        distances, indices = self.faiss_index_summaries.search(
            np.array([query_vector], dtype="float32"), k=k
        )

        top_files = []
        for idx in indices[0]:
            if 0 <= idx < len(self.paper_ids):
                top_files.append(self.paper_ids[idx])
            else:
                print(f"Warning: out-of-bound index {idx} in summary search.")
        return top_files

########################################
#     EMBEDDING INDEX AGENT (CHUNKS)   #
########################################

class EmbeddingIndexAgent:
    """
    Stage 2 index agent for chunk-level embeddings, using HNSWFlat for approximate search.
    """
    def __init__(self, model, index_file, chunks_file):
        self.model = model
        self.index_file = index_file
        self.chunks_file = chunks_file

    def load_or_create_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
            faiss_index = faiss.read_index(self.index_file)
            text_chunks = np.load(self.chunks_file, allow_pickle=True).tolist()
            return faiss_index, text_chunks
        return None, []

    def update_index(self, faiss_index, existing_text_chunks, new_doc_chunks):
        new_texts = [c["chunk"] for c in new_doc_chunks]
        new_embeddings = self.model.encode(new_texts, show_progress_bar=False)
        new_embeddings = np.array([x / np.linalg.norm(x) for x in new_embeddings], dtype="float32")

        if faiss_index is None:
            dimension = new_embeddings.shape[1]

            # Using HNSWFlat for chunk-level embeddings
            hnsw_index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the M parameter
            hnsw_index.hnsw.efConstruction = 80
            hnsw_index.hnsw.efSearch = 80
            hnsw_index.metric_type = faiss.METRIC_INNER_PRODUCT
            hnsw_index.add(new_embeddings)

            faiss_index = hnsw_index
        else:
            # Assuming HNSWFlat index
            faiss_index.add(new_embeddings)

        updated_text_chunks = existing_text_chunks + new_doc_chunks

        # Persist
        faiss.write_index(faiss_index, self.index_file)
        np.save(self.chunks_file, updated_text_chunks)

        return faiss_index, updated_text_chunks
