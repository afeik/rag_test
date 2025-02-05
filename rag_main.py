# app.py (Streamlit code)

import streamlit as st
import anthropic
import faiss
import numpy as np
import json
import os
from PIL import Image

from sentence_transformers import SentenceTransformer
from components.llm_agent import ClaudeAgent

# ---------------------------
# 1) Load / Initialize Globals
# ---------------------------

import pathlib

BASE_DIR = pathlib.Path(__file__).parent

INDEX_FILE = str(BASE_DIR / "faiss_index.bin")
TEXT_CHUNKS_FILE = str(BASE_DIR / "text_chunks.npy")
SUMMARY_INDEX_FILE = str(BASE_DIR / "faiss_index_summaries.bin")
METADATA_FILE = str(BASE_DIR / "metadata.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

# If you have GPU, set this to True to move Faiss indexes to GPU
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
    2) Use chunk-level index to get top chunk candidates (approx).
    3) Filter only chunks from those top papers.
    4) Sort the filtered chunks by similarity distance.
    5) Return a single text string as context.
    """
    if summary_ix is None or chunk_index is None or not paper_ids:
        return ""

    # 1) Encode + search summary-level
    q_emb = EMBEDDING_MODEL.encode([query], show_progress_bar=False)
    q_emb = np.array([v / np.linalg.norm(v) for v in q_emb], dtype="float32")

    dist_sum, idx_sum = summary_ix.search(q_emb, top_papers)

    relevant_papers = []
    if len(idx_sum) > 0:
        for idx in idx_sum[0]:
            if 0 <= idx < len(paper_ids):
                relevant_papers.append(paper_ids[idx])
    if not relevant_papers:
        print("No relevant papers found.")
        return ""

    print(f"Relevant Papers: {relevant_papers}")

    # 2) Chunk-level search (approx) for top big_k
    big_k = 20
    dist_chunks, idx_chunks = chunk_index.search(q_emb, big_k)

    # 3) Filter only chunks from relevant papers
    filtered = []
    for (chunk_idx, distance) in zip(idx_chunks[0], dist_chunks[0]):
        if chunk_idx < 0 or chunk_idx >= len(text_chunks):
            continue
        info = text_chunks[chunk_idx]
        # Use 'title' instead of 'paper_id'
        if info.get("title") in relevant_papers:
            filtered.append((distance, info))

    print(f"Filtered Chunks Count: {len(filtered)}")

    if not filtered:
        print("No chunks found after filtering.")
        return ""

    # 4) Sort the filtered results by ascending distance (higher similarity)
    sorted_filtered = sorted(filtered, key=lambda x: x[0])

    # 5) Take the top N
    top_results = sorted_filtered[:top_chunks]

    # Build final context
    context_parts = []
    for i, (dist, cinfo) in enumerate(top_results, 1):
        context_parts.append(
            f"**[{cinfo.get('title', 'Unknown')}]**\n{cinfo['chunk']}\n"
        )

    return "\n---\n".join(context_parts)


# ---------------------------
# 2) Page Configuration and Styling
# ---------------------------

# Set page configuration at the top, before any other Streamlit commands
st.set_page_config(layout="wide", page_title="Swiss Solar RAG Chatbot", page_icon="🔆")

# Custom CSS for sidebar text wrapping
st.markdown(
    """
    <style>
    /* Sidebar Text Wrapping */
    [data-testid="stSidebar"] .streamlit-expanderContent p {
        overflow-wrap: break-word !important;
        white-space: pre-wrap !important;
    }
    /* Ensure expander headers are not overlapping */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        white-space: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title("🔆 Swiss Solar RAG Chatbot")

    # Make sidebar wider
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 500px !important; /* Increased from 400px */
            max-width: 600px !important; /* Increased from 500px */
            background-color: #f0f2f6;
        }
        /* Optional: Adjust font size in sidebar */
        [data-testid="stSidebar"] .streamlit-expanderContent {
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "rag_context" not in st.session_state:
        st.session_state["rag_context"] = ""

    # Load indexes with a loading spinner
    with st.spinner("🔄 Loading indexes..."):
        chunk_ix, text_chunks, summary_ix, paper_ids = load_indexes()
    print(f"Loaded {chunk_ix.ntotal if chunk_ix else 0} chunks from the index.")

    if chunk_ix is None or summary_ix is None or not paper_ids:
        st.error("❌ Indexes not found. Please run your RAG indexing first.")
        st.stop()

    # Initialize Claude Agent
    claude_api_key = st.secrets["claude"]["claude_auth"]
    client = anthropic.Client(api_key=claude_api_key)
    claude_agent = ClaudeAgent(client)

    # Provide a welcome message if no conversation yet
    if len(st.session_state["messages"]) == 0:
        greeting = (
            "👋 Hello! I'm Claude, here to assist you with information about Switzerland's solar energy projects. "
            "Feel free to ask me anything related to photovoltaic systems, adoption rates, policies, and more!"
        )
        st.session_state["messages"].append({"role": "assistant", "content": greeting})

    # Chat interface container
    chat_container = st.container()

    # Display conversation
    with chat_container:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                # User message with inline styles for spacing
                st.markdown(
                    f"""
                    <div style="margin-bottom: 20px; display: flex; justify-content: flex-end;">
                        <div style="
                            background-color: #DCF8C6;
                            padding: 12px 16px;
                            border-radius: 15px;
                            max-width: 75%;
                            word-wrap: break-word;
                            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                        ">
                            {msg["content"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif msg["role"] == "assistant":
                # Assistant message with inline styles for spacing
                st.markdown(
                    f"""
                    <div style="margin-bottom: 20px; display: flex; justify-content: flex-start;">
                        <div style="
                            background-color: #F1F0F0;
                            padding: 12px 16px;
                            border-radius: 15px;
                            max-width: 75%;
                            word-wrap: break-word;
                            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                        ">
                            {msg["content"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Capture user input
    user_input = st.chat_input("💬 Ask a question or share your thoughts...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with chat_container:
            # Display user message with inline styles for spacing
            st.markdown(
                f"""
                <div style="margin-bottom: 20px; display: flex; justify-content: flex-end;">
                    <div style="
                        background-color: #DCF8C6;
                        padding: 12px 16px;
                        border-radius: 15px;
                        max-width: 75%;
                        word-wrap: break-word;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                    ">
                        {user_input}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display typing indicator with inline styles
            st.markdown(
                """
                <div style="margin-bottom: 20px; display: flex; justify-content: flex-start;">
                    <div style="
                        background-color: #F1F0F0;
                        padding: 12px 16px;
                        border-radius: 15px;
                        max-width: 75%;
                        word-wrap: break-word;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                        display: flex;
                        align-items: center;
                        gap: 5px;
                        font-style: italic;
                        color: #555;
                    ">
                        <div style="
                            height: 8px;
                            width: 8px;
                            background-color: #bbb;
                            border-radius: 50%;
                            display: inline-block;
                            animation: typing 1.4s infinite;
                        "></div>
                        <div style="
                            height: 8px;
                            width: 8px;
                            background-color: #bbb;
                            border-radius: 50%;
                            display: inline-block;
                            animation: typing 1.4s infinite;
                            animation-delay: 0.2s;
                        "></div>
                        <div style="
                            height: 8px;
                            width: 8px;
                            background-color: #bbb;
                            border-radius: 50%;
                            display: inline-block;
                            animation: typing 1.4s infinite;
                            animation-delay: 0.4s;
                        "></div>
                        <span>Claude is typing...</span>
                    </div>
                </div>

                <style>
                @keyframes typing {
                    0% { opacity: 0.2; }
                    20% { opacity: 1; }
                    100% { opacity: 0.2; }
                }
                </style>
                """,
                unsafe_allow_html=True
            )

        # Perform RAG retrieval and Claude call without spinner
        rag_context = retrieve_rag_context(
            user_input,
            summary_ix,
            paper_ids,
            chunk_ix,
            text_chunks,
        )

        # Call Claude
        try:
            answer = claude_agent.call_claude(st.session_state["messages"], rag_context)
        except Exception as e:
            answer = f"❌ Error generating response: {e}"
            print(f"Error calling Claude: {e}")

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.session_state["rag_context"] = rag_context

        # Update chat with assistant's response by rerunning the app
        st.rerun()

    # Sidebar: Show RAG context
    with st.sidebar.expander("🗃️ Current RAG Context", expanded=False):
        if "rag_context" in st.session_state and st.session_state["rag_context"]:
            st.write(st.session_state["rag_context"])
        else:
            st.info("🔍 Context will appear here after your first question.")

    # Show known paper IDs
    with st.sidebar.expander("📄 Paper Titles", expanded=False):
        if paper_ids:
            for pid in paper_ids:
                st.write(f"- {pid}")
        else:
            st.info("No paper titles available.")

    # Footer (optional)
    st.markdown(
        """
        <style>
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
