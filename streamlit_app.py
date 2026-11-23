"""Web UI for the AI Research Assistant.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import hashlib
import os
from datetime import datetime

import streamlit as st
from pypdf import PdfReader

from rag_pipeline import DEFAULT_PDF, PROJECT_ROOT, AIResearchAssistant, load_embeddings

UPLOAD_DIR = PROJECT_ROOT / "data" / "pdfs" / "uploads"
SUGGESTED_QUESTIONS = [
    "Summarize the main topics of this document",
    "What are the key definitions I should know?",
    "Explain the most important concept in simple terms",
]

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="centered")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the AI model (first launch only)...")
def get_embeddings():
    return load_embeddings()


@st.cache_resource(show_spinner=False)
def get_assistant(pdf_path: str, content_hash: str) -> AIResearchAssistant:
    # content_hash is part of the cache key so an edited file gets a fresh assistant
    assistant = AIResearchAssistant(pdf_path=pdf_path, embeddings=get_embeddings())
    assistant.load_vector_store()  # instant if this PDF was indexed before
    return assistant


@st.cache_data(show_spinner=False)
def page_count(pdf_path: str, content_hash: str) -> int:
    return len(PdfReader(pdf_path).pages)


def file_hash(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def friendly_error(e: Exception) -> str:
    """Turn API/library exceptions into messages a user can act on."""
    import groq

    if isinstance(e, groq.AuthenticationError):
        return "🔑 Your Groq API key was rejected. Check that it's correct and paste it again."
    if isinstance(e, groq.RateLimitError):
        return "⏳ Groq's rate limit was reached. Wait a few seconds and try again."
    if isinstance(e, groq.APIConnectionError):
        return "🌐 Couldn't reach Groq. Check your internet connection and try again."
    if isinstance(e, groq.NotFoundError):
        return ("🤖 The configured AI model isn't available. Set `GROQ_MODEL` in your "
                ".env file to a current Groq model.")
    return f"Something went wrong: {e}"


def open_document(path: str):
    st.session_state.pdf_path = path
    st.session_state.chats.setdefault(path, [])


def chat_as_markdown(name: str, messages: list) -> str:
    lines = [f"# Chat about {name}", f"_Exported {datetime.now():%Y-%m-%d %H:%M}_", ""]
    for m in messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines += [f"**{who}:** {m['content']}"]
        if m.get("pages"):
            lines.append(f"_Sources: pages {', '.join(map(str, m['pages']))}_")
        lines.append("")
    return "\n".join(lines)


def show_sources(docs):
    with st.expander("📑 View source passages"):
        for d in docs:
            st.markdown(f"**Page {d.metadata.get('page', '?')}**")
            text = d.page_content.strip()
            st.caption(text[:600] + ("…" if len(text) > 600 else ""))


st.session_state.setdefault("pdf_path", None)
st.session_state.setdefault("chats", {})  # pdf path -> list of messages
has_key = bool(os.getenv("GROQ_API_KEY"))


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Research Assistant")

    if st.session_state.pdf_path:
        path = st.session_state.pdf_path
        assistant_for_stats = get_assistant(path, file_hash(path))
        with st.container(border=True):
            st.markdown(f"**📄 {os.path.basename(path)}**")
            info = f"{page_count(path, file_hash(path))} pages"
            if assistant_for_stats.is_ready:
                info += f" · {assistant_for_stats.vector_store._collection.count()} chunks indexed"
            st.caption(info)
        if st.button("↩️ Change document", use_container_width=True):
            st.session_state.pdf_path = None
            st.rerun()

        messages = st.session_state.chats.get(path, [])
        if messages:
            st.download_button(
                "⬇️ Download chat", chat_as_markdown(os.path.basename(path), messages),
                file_name=f"chat_{os.path.splitext(os.path.basename(path))[0]}.md",
                mime="text/markdown", use_container_width=True,
            )
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chats[path] = []
                st.rerun()

    st.divider()
    if has_key:
        st.success("AI connected", icon="✅")
    else:
        st.warning("AI not connected: add your Groq API key", icon="🔑")

    with st.expander("⚙️ Advanced settings"):
        k = st.slider(
            "Passages per answer", 1, 10, 4,
            help="How many of the most relevant passages the AI reads before answering. "
                 "More = broader answers, but slower.",
        )

    with st.expander("ℹ️ How does it work?"):
        st.markdown(
            "1. Your PDF is split into small passages.\n"
            "2. Each passage is converted into a *meaning vector* (embedding).\n"
            "3. Your question is matched to the passages closest in **meaning** "
            "(semantic search), not just keywords.\n"
            "4. The AI answers using **only** those passages and cites the pages."
        )


# ----------------------------------------------------------------------
# Landing page
# ----------------------------------------------------------------------
if not st.session_state.pdf_path:
    st.title("📚 AI Research Assistant")
    st.markdown("#### Ask questions about any PDF and get answers with page citations.")
    st.write(
        "Upload lecture notes, a research paper or a report, then chat with it. "
        "Answers come **only from your document**, and each one shows the pages it used, "
        "so you can always check the source."
    )

    uploaded = st.file_uploader(
        "**Upload a PDF to get started**", type="pdf",
        help="PDFs with selectable text work best. Scanned images can't be read.",
    )
    if uploaded is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOAD_DIR / os.path.basename(uploaded.name)
        path.write_bytes(uploaded.getvalue())
        open_document(str(path))
        st.rerun()

    if DEFAULT_PDF.exists():
        c1, c2 = st.columns([1, 2], vertical_alignment="center")
        if c1.button("✨ Try the sample notes", type="primary", use_container_width=True):
            open_document(str(DEFAULT_PDF))
            st.rerun()
        c2.caption("No PDF handy? Explore the included Machine Learning study notes.")

    recent = [p for p in st.session_state.chats if os.path.exists(p)]
    if recent:
        st.markdown("##### Continue where you left off")
        for p in recent:
            n = len(st.session_state.chats[p]) // 2
            if st.button(f"📄 {os.path.basename(p)} · {n} question(s)", key=f"recent_{p}"):
                open_document(p)
                st.rerun()

    st.divider()
    st.markdown("##### How it works")
    cols = st.columns(3)
    steps = [
        ("📤", "Upload", "Add your PDF. It's indexed once, then reused instantly."),
        ("🔎", "Ask", "Your question is matched to passages by *meaning*, not keywords."),
        ("✅", "Verify", "Get a clear answer with page numbers and the exact source passages."),
    ]
    for col, (icon, title, text) in zip(cols, steps):
        with col, st.container(border=True):
            st.markdown(f"### {icon}\n**{title}**")
            st.caption(text)
    st.stop()


# ----------------------------------------------------------------------
# Chat page
# ----------------------------------------------------------------------
pdf_path = st.session_state.pdf_path
name = os.path.basename(pdf_path)
messages = st.session_state.chats.setdefault(pdf_path, [])

try:
    assistant = get_assistant(pdf_path, file_hash(pdf_path))
except Exception as e:
    st.error(f"Could not open the PDF: {e}")
    st.stop()
assistant.k = k

# Index on first use, with a real progress bar
if not assistant.is_ready:
    with st.status(f"Preparing **{name}** (first time only)...", expanded=True) as status:
        bar = st.progress(0.0, text="Starting...")
        try:
            assistant.ingest_pdf(progress=lambda f, msg: bar.progress(min(f, 1.0), text=msg))
        except Exception as e:
            assistant.vector_store = None
            status.update(label="Couldn't process this PDF", state="error")
            st.error(str(e))
            st.stop()
        status.update(label=f"**{name}** is ready!", state="complete", expanded=False)
    st.rerun()  # refresh the sidebar stats

st.title("💬 Ask your document")
st.caption(f"Answers are based only on **{name}**.")

if not has_key:
    with st.container(border=True):
        st.markdown("**🔑 Connect the AI to start asking questions**")
        st.caption(
            "Paste a free Groq API key from [console.groq.com/keys](https://console.groq.com/keys). "
            "To skip this step next time, save it as `GROQ_API_KEY` in the project's `.env` file."
        )
        key = st.text_input("Groq API key", type="password", placeholder="gsk_...",
                            label_visibility="collapsed")
        if key:
            os.environ["GROQ_API_KEY"] = key.strip()
            assistant._llm = None
            st.rerun()

# Chat history
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("pages"):
            st.caption(f"📄 Sources: pages {', '.join(map(str, msg['pages']))}")
        if msg.get("sources"):
            show_sources(msg["sources"])

# Suggested questions for an empty chat
pending = None
if not messages:
    st.markdown("**Not sure where to start? Try one of these:**")
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if st.button(q, key=f"suggest_{i}", disabled=not has_key, use_container_width=True):
            pending = q

typed = st.chat_input(
    "Ask a question about your document..." if has_key else "Add your API key above to start",
    disabled=not has_key,
)
question = typed or pending

if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your document..."):
                result = assistant.ask_stream(question)
            answer = st.write_stream(result["answer"])
            if result["pages"]:
                st.caption(f"📄 Sources: pages {', '.join(map(str, result['pages']))}")
            show_sources(result["source_docs"])
            messages.append({
                "role": "assistant", "content": answer,
                "pages": result["pages"], "sources": result["source_docs"],
            })
        except Exception as e:
            messages.pop()  # drop the unanswered question so the user can retry
            st.error(friendly_error(e))
            pending = None  # keep the error visible (no rerun)
    if pending:
        st.rerun()  # hide the suggestion buttons once the chat has started
