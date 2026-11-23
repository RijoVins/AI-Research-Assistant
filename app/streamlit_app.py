"""Web UI for the AI Research Assistant.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import hashlib
import os

import streamlit as st

from rag_pipeline import DEFAULT_PDF, PROJECT_ROOT, AIResearchAssistant, load_embeddings

UPLOAD_DIR = PROJECT_ROOT / "data" / "pdfs" / "uploads"

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------
# Cached heavy objects (survive Streamlit reruns)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return load_embeddings()


@st.cache_resource(show_spinner=False)
def get_assistant(pdf_path: str, content_hash: str, k: int) -> AIResearchAssistant:
    # content_hash is part of the cache key so a changed file gets a fresh assistant
    assistant = AIResearchAssistant(pdf_path=pdf_path, k=k, embeddings=get_embeddings())
    assistant.setup()
    return assistant


def file_hash(path) -> str:
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


# ----------------------------------------------------------------------
# Sidebar: document + settings
# ----------------------------------------------------------------------
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
    st.session_state.messages = []

with st.sidebar:
    st.header("📄 Your document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    # Only react to a *new* upload, so "Use sample notes" isn't overridden on rerun
    if uploaded is not None and uploaded.file_id != st.session_state.get("last_upload"):
        st.session_state.last_upload = uploaded.file_id
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOAD_DIR / os.path.basename(uploaded.name)
        path.write_bytes(uploaded.getvalue())
        st.session_state.pdf_path = str(path)
        st.session_state.messages = []

    if DEFAULT_PDF.exists() and st.button("Use sample notes (Notes.pdf)", use_container_width=True):
        st.session_state.pdf_path = str(DEFAULT_PDF)
        st.session_state.messages = []

    st.divider()
    st.header("⚙️ Settings")
    k = st.slider("Chunks to retrieve (k)", 1, 10, 4,
                  help="How many of the most similar chunks are sent to the LLM.")
    if not os.getenv("GROQ_API_KEY"):
        key = st.text_input("Groq API key", type="password",
                            help="Or put GROQ_API_KEY in the .env file.")
        if key:
            os.environ["GROQ_API_KEY"] = key
            st.rerun()
    else:
        st.success("Groq API key loaded", icon="🔑")

    if st.session_state.messages and st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------------------------------------
# Landing page (no document selected yet)
# ----------------------------------------------------------------------
if not st.session_state.pdf_path:
    st.title("📚 AI Research Assistant")
    st.subheader("Chat with your PDFs. Get answers grounded in your own documents.")
    st.write(
        "Upload a PDF (lecture notes, a research paper, a report) and ask questions in plain "
        "language. The assistant finds the most relevant passages using **semantic search** "
        "and answers using only those passages, with page citations, so you can verify every answer."
    )

    st.markdown("### How it works")
    cols = st.columns(4)
    steps = [
        ("1️⃣ Upload", "Add a PDF from the sidebar, or try the sample notes."),
        ("2️⃣ Index", "The text is split into chunks and turned into embeddings stored in Chroma."),
        ("3️⃣ Search", "Your question is matched to the chunks closest in *meaning*, not just keywords."),
        ("4️⃣ Answer", "An LLM (Groq) writes the answer from those chunks and cites the pages."),
    ]
    for col, (title, text) in zip(cols, steps):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)

    st.info("👈 **Get started:** upload a PDF in the sidebar, or click *Use sample notes*.")
    st.stop()


# ----------------------------------------------------------------------
# Chat page
# ----------------------------------------------------------------------
pdf_path = st.session_state.pdf_path
st.title("📚 AI Research Assistant")
st.caption(f"Chatting with **{os.path.basename(pdf_path)}**")

try:
    with st.spinner("Indexing the PDF (first time only, this can take a few minutes)..."):
        assistant = get_assistant(pdf_path, file_hash(pdf_path), k)
except Exception as e:
    st.error(f"Could not process the PDF: {e}")
    st.stop()


def show_sources(docs):
    with st.expander(f"Sources ({len(docs)} chunks)"):
        for d in docs:
            st.markdown(f"**Page {d.metadata.get('page', '?')}**")
            st.caption(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

if question := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your document..."):
                result = assistant.ask(question)
            answer = result["answer"]
            if result["pages"]:
                answer += f"\n\n*Sources: pages {', '.join(map(str, result['pages']))}*"
            st.markdown(answer)
            show_sources(result["source_docs"])
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": result["source_docs"]}
            )
        except Exception as e:
            st.error(f"Error: {e}")
