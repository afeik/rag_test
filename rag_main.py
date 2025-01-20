import streamlit as st
import anthropic
import faiss
import numpy as np
import json
import os

import faiss
import numpy as np


from sentence_transformers import SentenceTransformer
from components.llm_agent import ClaudeAgent

# ---------------------------
# 1) Load / Initialize Globals
# ---------------------------
INDEX_FILE = "./faiss_index.bin"
TEXT_CHUNKS_FILE = "./text_chunks.npy"
SUMMARY_INDEX_FILE = "./faiss_index_summaries.bin"
METADATA_FILE = "./metadata.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

# If you have GPU, set this to True to load Faiss indexes to GPU
USE_GPU = False

def load_indexes():
    """Load chunk-level and summary-level indexes plus metadata. 
       Convert them to a GPU index if USE_GPU=True and a GPU is available.
    """
    chunk_ix, text_chunks = None, []
    summary_ix, paper_ids = None, []

    if os.path.exists(INDEX_FILE) and os.path.exists(TEXT_CHUNKS_FILE):
        chunk_ix = faiss.read_index(INDEX_FILE)
        text_chunks = np.load(TEXT_CHUNKS_FILE, allow_pickle=True).tolist()

    if os.path.exists(SUMMARY_INDEX_FILE) and os.path.exists(METADATA_FILE):
        summary_ix = faiss.read_index(SUMMARY_INDEX_FILE)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        paper_ids = metadata.get("_paper_id_order", [])

    # --- Move to GPU if desired and available ---
    if USE_GPU:
        try:
            res = faiss.StandardGpuResources()
            if chunk_ix is not None:
                chunk_ix = faiss.index_cpu_to_gpu(res, 0, chunk_ix)
            if summary_ix is not None:
                summary_ix = faiss.index_cpu_to_gpu(res, 0, summary_ix)
        except Exception as e:
            print("Could not move indexes to GPU:", e)

    return chunk_ix, text_chunks, summary_ix, paper_ids


def retrieve_rag_context(
    query,
    summary_ix,
    paper_ids,
    chunk_index,
    text_chunks,
    top_papers=2,
    top_chunks=3
):
    """
    1) Use summary index to find top relevant papers for the query.
    2) Use chunk-level index to get top chunk candidates.
    3) Filter to keep only chunks from those top papers.
    4) Return a single text string as context.
    """
    if summary_ix is None or chunk_index is None or not paper_ids:
        return ""

    # 1) Encode + search summary-level
    q_emb = EMBEDDING_MODEL.encode([query], show_progress_bar=False)
    # (Optional) If your Faiss index is built with normalized vectors or IP, 
    # you might skip this manual normalization. Ensure consistent usage.
    q_emb = np.array([v / np.linalg.norm(v) for v in q_emb], dtype="float32")

    # Top X relevant papers
    dist_sum, idx_sum = summary_ix.search(q_emb, top_papers)

    relevant_papers = []
    if len(idx_sum) > 0:
        for idx in idx_sum[0]:
            if 0 <= idx < len(paper_ids):
                relevant_papers.append(paper_ids[idx])
    if not relevant_papers:
        return ""

    # 2) Chunk-level search: retrieve fewer candidates to speed up
    big_k = 20  # Instead of 50 or 100, smaller number speeds up
    dist_chunks, idx_chunks = chunk_index.search(q_emb, big_k)

    # 3) Filter to keep only chunks belonging to relevant papers
    filtered = []
    for (chunk_idx, distance) in zip(idx_chunks[0], dist_chunks[0]):
        if chunk_idx < 0 or chunk_idx >= len(text_chunks):
            continue
        info = text_chunks[chunk_idx]
        # 'title' in info is the paper's ID / Title
        if info["title"] in relevant_papers:
            filtered.append((distance, info))

    # 4) Sort by distance descending (or ascending if L2). 
    #    If it's IP (dot product), you might want descending; if L2, ascending.
    #    Adjust accordingly. Let's assume higher better for IP.
    filtered.sort(key=lambda x: x[0], reverse=True)

    top_results = filtered[:top_chunks]
    if not top_results:
        return ""

    # Build context (Markdown style)
    context_parts = []
    for i, (dist, cinfo) in enumerate(top_results, 1):
        context_parts.append(
            f"**[{cinfo.get('title', 'Unknown')}]**\n{cinfo['chunk']}\n"
        )

    return "\n---\n".join(context_parts)


# ----------------------------
# 2) Streamlit Multi-Turn Chat
# ----------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Swiss Solar RAG Chatbot")

    # Make sidebar a bit bigger
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 320px;
            max-width: 380px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Loading indexes..."):
        chunk_ix, text_chunks, summary_ix, paper_ids = load_indexes()

    #if chunk_ix is None or summary_ix is None or not paper_ids:
    #    st.error("Indexes not found. Please run your RAG indexing first.")
    #    return

    claude_api_key = st.secrets["claude"]["claude_auth"]
    client = anthropic.Client(api_key=claude_api_key)
    claude_agent = ClaudeAgent(client)

    # Initialize conversation state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Provide a welcome message if no conversation yet
    if len(st.session_state["messages"]) == 0:
        greeting = (
            "Hello! I'm Claude, supporting Switzerland's solar energy project. "
            "Ask me anything about photovoltaic systems, adoption, or related topics!"
        )
        st.session_state["messages"].append({"role": "assistant", "content": greeting})

    # 1) Capture user input
    user_input = st.chat_input("Ask a question or share your thoughts...")

    if user_input:
        # Add the new user message
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # RAG retrieval
        rag_context = retrieve_rag_context(
            user_input,
            summary_ix,
            paper_ids,
            chunk_ix,
            text_chunks,
        )

        # Call Claude
        answer = claude_agent.call_claude(st.session_state["messages"], rag_context)

        # Save the assistant's reply
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.session_state["rag_context"] = rag_context

    # 2) Sidebar: Show the current RAG context
    with st.sidebar.expander("Current RAG Context", expanded=False):
        if "rag_context" in st.session_state:
            st.write(st.session_state["rag_context"])
        else:
            st.info("Context will appear here after your first question.")

    # 3) Show known paper IDs
    st.sidebar.markdown("**Paper Titles**")
    for pid in paper_ids:
        st.sidebar.write(f"- {pid}")

    # 4) Display the conversation
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


if __name__ == "__main__":
    main()
