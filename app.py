"""
app.py
------
PRESENTATION LAYER ONLY. All data processing, embedding, retrieval, and
LLM logic lives untouched in utils/ - this file only decides how things
LOOK and are LAID OUT. Every call into utils/ (extract_text_from_multiple_pdfs,
VectorStore, answer_question) is unchanged from the functional version.

Layout: Streamlit's sidebar is used as the "left utility zone" (upload +
file list + embeddings demo), and the main area is the full-width chat
theater on the right - this maps cleanly onto Streamlit's native
architecture without fighting the framework.
"""

import hashlib

import streamlit as st
from utils.pdf_utils import PDFExtractionError, extract_text_from_multiple_pdfs
from utils.vector_store import VectorStore
from utils.qa import answer_question, select_context

st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄", layout="wide")


# =================================================================
# DESIGN TOKENS - single source of truth for the whole theme
# =================================================================
BG_PRIMARY = "#0B0F19"       # deep midnight obsidian
BG_CARD = "#161B22"          # soft slate charcoal
BG_USER_BUBBLE = "#242938"   # dark mid-gray for user pills
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#E2E8F0"
TEXT_MUTED = "#7B8394"
ACCENT = "#3B82F6"           # electric cobalt blue
BORDER = "#242938"


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}

        /* ---- Base canvas ---- */
        .stApp {{
            background-color: {BG_PRIMARY};
        }}
        [data-testid="stHeader"] {{
            background-color: transparent;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-left: clamp(1rem, 3vw, 3rem);
            padding-right: clamp(1rem, 3vw, 3rem);
            max-width: 1440px;
        }}
        [data-testid="stToolbar"] {{
            visibility: hidden;
        }}

        /* ---- Sidebar (left utility zone) ---- */
        [data-testid="stSidebar"] {{
            background-color: {BG_CARD};
            border-right: 1px solid {BORDER};
        }}
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: {TEXT_PRIMARY};
            font-weight: 600;
        }}
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{
            color: {TEXT_SECONDARY};
        }}

        /* ---- File uploader dropzone ---- */
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {BG_PRIMARY};
            border: 1.5px dashed {BORDER};
            border-radius: 14px;
            transition: border-color 0.2s ease, background-color 0.2s ease;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {ACCENT};
            background-color: rgba(59, 130, 246, 0.06);
        }}

        /* ---- Headers ---- */
        h1, h2, h3 {{
            color: {TEXT_PRIMARY} !important;
            letter-spacing: -0.02em;
        }}
        p, span, div {{
            color: {TEXT_SECONDARY};
        }}

        /* ---- Buttons ---- */
        .stButton button, .stDownloadButton button {{
            background-color: {ACCENT};
            color: {TEXT_PRIMARY};
            border: none;
            border-radius: 10px;
            font-weight: 500;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .stButton button:hover, .stDownloadButton button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35);
        }}

        /* ---- Alerts / info / success boxes - de-corporatized ---- */
        [data-testid="stAlert"] {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-left: 3px solid {ACCENT};
            border-radius: 10px;
            color: {TEXT_SECONDARY};
        }}

        /* ---- Expanders ---- */
        [data-testid="stExpander"] {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 12px;
        }}

        /* ---- Chat messages: fade-in-up + bubble treatment ---- */
        @keyframes fadeInUp {{
            from {{ transform: translateY(8px); opacity: 0; }}
            to   {{ transform: translateY(0);    opacity: 1; }}
        }}
        [data-testid="stChatMessage"] {{
            animation: fadeInUp 0.35s ease-out both;
            background-color: transparent;
            padding: 0.25rem 0;
            max-width: min(88%, 980px);
        }}
        /* Messages alternate user -> assistant in strict render order,
           so nth-of-type(odd/even) reliably maps to role without needing
           fragile internal selectors. */
        [data-testid="stChatMessage"]:nth-of-type(odd) {{
            margin-left: auto;
            flex-direction: row-reverse;
        }}
        [data-testid="stChatMessage"]:nth-of-type(odd) [data-testid="stChatMessageContent"] {{
            background-color: {BG_USER_BUBBLE};
            border-radius: 18px 18px 4px 18px;
            padding: 0.75rem 1.1rem;
        }}
        [data-testid="stChatMessage"]:nth-of-type(even) {{
            margin-right: auto;
        }}
        [data-testid="stChatMessage"]:nth-of-type(even) [data-testid="stChatMessageContent"] {{
            background-color: {BG_CARD};
            border-left: 3px solid {ACCENT};
            border-radius: 4px 18px 18px 18px;
            padding: 0.75rem 1.1rem;
        }}

        /* ---- Chat input: locked-bottom pill ---- */
        [data-testid="stChatInput"] {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 999px;
            padding: 0.2rem 0.5rem;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        [data-testid="stChatInput"]:focus-within {{
            border-color: {ACCENT};
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }}
        [data-testid="stChatInput"] textarea {{
            color: {TEXT_PRIMARY} !important;
        }}

        /* ---- Misc cleanup ---- */
        #MainMenu, footer {{ visibility: hidden; }}

        @media (max-width: 768px) {{
            .block-container {{ padding-top: 1rem; }}
            [data-testid="stChatMessage"] {{ max-width: 100%; }}
            [data-testid="stChatMessageContent"] {{ padding: 0.7rem 0.85rem !important; }}
            [data-testid="stSidebar"] {{ min-width: min(86vw, 320px); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_logo_header():
    """
    Minimal line-art SVG: a document/folder silhouette intersecting with
    a chat bubble node - built as a single inline SVG so it needs no
    external image asset.
    """
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.25rem;">
            <svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M6 6H18L22 10H28V26C28 27.1 27.1 28 26 28H6C4.9 28 4 27.1 4 26V8C4 6.9 4.9 6 6 6Z"
                      stroke="{TEXT_PRIMARY}" stroke-width="1.6" fill="none" stroke-linejoin="round"/>
                <path d="M18 18C18 15.8 20.2 14 23 14C25.8 14 28 15.8 28 18C28 19.5 27.2 20.8 25.9 21.6L26.3 24L23.8 22.6C23.5 22.65 23.25 22.65 23 22.65C20.2 22.65 18 20.85 18 18Z"
                      fill="{ACCENT}" stroke="{ACCENT}" stroke-width="1.2" stroke-linejoin="round"/>
            </svg>
            <span style="font-size:1.15rem; font-weight:600; color:{TEXT_PRIMARY}; letter-spacing:-0.02em;">
                PDF Chat Assistant
            </span>
        </div>
        <p style="color:{TEXT_MUTED}; font-size:0.9rem; margin-top:0; margin-bottom:1.5rem;">
            Ask questions grounded in your uploaded documents.
        </p>
        """,
        unsafe_allow_html=True,
    )


inject_css()
render_logo_header()

# ---------------------------------------------------------------
# SESSION STATE (unchanged from functional version)
# ---------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


def render_sources(retrieved_chunks):
    if not retrieved_chunks:
        return
    pages_by_source = {}
    for chunk in retrieved_chunks:
        pages_by_source.setdefault(chunk["source"], set()).add(chunk["page"])
    with st.expander("Sources used for this answer"):
        for source, pages in pages_by_source.items():
            page_list = ", ".join(str(p) for p in sorted(pages))
            label = "page" if len(pages) == 1 else "pages"
            st.markdown(f"📄 **{source}** — {label} {page_list}")


# =================================================================
# LEFT ZONE (sidebar): upload, file list, embeddings demo
# =================================================================
with st.sidebar:
    st.markdown("### Documents")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        current_files = [(f.name, f.size, hashlib.sha256(f.getvalue()).hexdigest()) for f in uploaded_files]

        if current_files != st.session_state.processed_files:
            with st.spinner("Extracting text and building embeddings..."):
                try:
                    documents = extract_text_from_multiple_pdfs(uploaded_files)
                    store = VectorStore()
                    store.build_from_documents(documents)
                    if store.index is None:
                        raise PDFExtractionError("No text chunks could be created from these files.")
                    st.session_state.vector_store = store
                    st.session_state.processed_files = current_files
                    st.session_state.chat_history = []
                except PDFExtractionError as error:
                    st.session_state.vector_store = None
                    st.error(str(error))
                    st.stop()
            st.success(f"{len(uploaded_files)} file(s) ready")
        else:
            st.markdown(
                f'<p style="color:{TEXT_MUTED}; font-size:0.85rem;">'
                f'{len(current_files)} file(s) loaded</p>',
                unsafe_allow_html=True,
            )

        for name, _, _ in current_files:
            st.markdown(
                f'<div style="background-color:{BG_PRIMARY}; border:1px solid {BORDER}; '
                f'border-radius:8px; padding:0.5rem 0.75rem; margin-bottom:0.4rem; '
                f'font-size:0.85rem; color:{TEXT_SECONDARY};">📄 {name}</div>',
                unsafe_allow_html=True,
            )
    elif st.session_state.processed_files:
        st.session_state.vector_store = None
        st.session_state.processed_files = []
        st.session_state.chat_history = []

    st.markdown("---")

    with st.expander("🔍 View Embeddings (demo / explanation)"):
        store = st.session_state.vector_store
        if store is None or store.index is None:
            st.write("Upload a PDF first.")
        else:
            num_chunks = store.index.ntotal
            st.write(
                f"Split into **{num_chunks} chunks**, each a "
                f"**{store.embedding_dim}-number vector**."
            )
            sample_chunk = store.chunk_metadata[0]
            all_vectors = store.get_all_embeddings()
            st.markdown("**Sample chunk text:**")
            st.text(sample_chunk["text"][:150] + "...")
            st.markdown(f"**First 15 of {store.embedding_dim} numbers:**")
            st.code(str(all_vectors[0][:15].round(4).tolist()))

            if num_chunks >= 3:
                st.markdown("**2D map (similar meaning = closer):**")
                from sklearn.decomposition import PCA
                import matplotlib.pyplot as plt

                reduced = PCA(n_components=2).fit_transform(all_vectors)
                sources = [m["source"] for m in store.chunk_metadata]
                unique_sources = list(set(sources))
                colors = plt.cm.tab10(range(len(unique_sources)))

                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor(BG_CARD)
                ax.set_facecolor(BG_CARD)
                for i, source in enumerate(unique_sources):
                    indices = [j for j, s in enumerate(sources) if s == source]
                    ax.scatter(
                        reduced[indices, 0], reduced[indices, 1],
                        label=source, color=colors[i], alpha=0.8, s=45,
                    )
                ax.tick_params(colors=TEXT_MUTED, labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color(BORDER)
                ax.legend(fontsize=6, facecolor=BG_CARD, labelcolor=TEXT_SECONDARY)
                st.pyplot(fig)


# =================================================================
# RIGHT ZONE (main area): the chat theater
# =================================================================
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        render_sources(entry["sources"])

question = st.chat_input("Ask a question about your uploaded PDF(s)...")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload at least one PDF first.")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            retrieved_chunks, mode = select_context(
                question, st.session_state.vector_store, st.session_state.chat_history
            )
            status = (
                "Using the previous answer's sources..."
                if mode == "follow_up"
                else "Generating response..."
                if mode == "acknowledgement"
                else "Searching documents and generating answer..."
            )
            with st.spinner(status):
                answer = answer_question(
                    question, retrieved_chunks, chat_history=st.session_state.chat_history
                )
            st.write(answer)
            render_sources(retrieved_chunks)

        st.session_state.chat_history.append(
            {"question": question, "answer": answer, "sources": retrieved_chunks}
        )
