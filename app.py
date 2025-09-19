import os
import uuid
from typing import List, Dict, Optional

import streamlit as st
from dotenv import load_dotenv

# LangChain core
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings as ChromaSettings  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Embeddings
try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
    HAS_LANGCHAIN_OPENAI = True
except Exception:
    HAS_LANGCHAIN_OPENAI = False

from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
    HAS_GOOGLE_EMBED = True
except Exception:
    HAS_GOOGLE_EMBED = False

# LLMs
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    HAS_CHAT_OPENAI = True
except Exception:
    HAS_CHAT_OPENAI = False

from langchain_community.llms import HuggingFaceHub
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    HAS_CHAT_GEMINI = True
except Exception:
    HAS_CHAT_GEMINI = False

from langchain.callbacks.base import BaseCallbackHandler


DB_DIR = os.path.join(".", "chroma_db")
DOCS_DIR = os.path.join(".", "docs")


def get_embeddings(api_key: Optional[str], gemini_key: Optional[str] = None):
    if api_key and HAS_LANGCHAIN_OPENAI:
        return OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    if gemini_key and HAS_GOOGLE_EMBED:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm(model_name: str, api_key: Optional[str], gemini_key: Optional[str]):
    # Default: OpenAI chat models; Fallback: HuggingFaceHub Inference (requires HF token if using hosted),
    # but user requested free tools only; we can default to a local small model via transformers pipeline through langchain if no OpenAI.
    if api_key and HAS_CHAT_OPENAI and model_name in {"gpt-3.5-turbo", "gpt-4"}:
        return ChatOpenAI(model_name=model_name, temperature=0.2, api_key=api_key)

    if gemini_key and HAS_CHAT_GEMINI and model_name.startswith("gemini"):
        # Common: "gemini-1.5-pro", "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key, temperature=0.2)

    # Fallback: local pipeline via sentence-transformers as encoder and a small causal LM for generation
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    local_model = "distilbert/distilgpt2"  # small and free to download
    tokenizer = AutoTokenizer.from_pretrained(local_model)
    model = AutoModelForCausalLM.from_pretrained(local_model)
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=text_gen)


def ensure_dirs():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)


def load_or_create_vectordb(embeddings):
    client_settings = ChromaSettings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=DB_DIR,
    )
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="documents",
        client_settings=client_settings,
    )


def add_uploaded_files_to_chroma(files, embeddings):
    docs: List[Document] = []
    for uf in files:
        filename = uf.name
        bytes_data = uf.read()
        # Save to docs dir for persistence
        saved_path = os.path.join(DOCS_DIR, filename)
        with open(saved_path, "wb") as f:
            f.write(bytes_data)

        # Load as text for indexing
        if filename.lower().endswith(".pdf"):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(saved_path)
            docs.extend(loader.load())
        else:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(saved_path, encoding="utf-8")
            docs.extend(loader.load())

    if not docs:
        return 0

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = splitter.split_documents(docs)
    vectordb = load_or_create_vectordb(embeddings)
    vectordb.add_documents(splits)
    vectordb.persist()
    return len(splits)


def retrieve_context(query: str, embeddings, k: int = 4):
    vectordb = load_or_create_vectordb(embeddings)
    return vectordb.similarity_search(query, k=k)


def format_sources(docs: List[Document]) -> str:
    lines = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "Unknown"
        if src not in seen:
            seen.add(src)
            lines.append(f"- {os.path.basename(src)}")
    return "\n".join(lines) if lines else "No sources found."


def build_prompt(user_query: str, context_docs: List[Document]) -> str:
    context_text = "\n\n".join([d.page_content[:1500] for d in context_docs])
    prompt = (
        "You are a helpful knowledge assistant. Use the provided context to answer the user's question.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {user_query}\n\n"
        "Answer:"
    )
    return prompt


def generate_response(llm, prompt: str) -> str:
    try:
        # ChatOpenAI supports invoke with a list of messages; for text LLMs we pass string
        if hasattr(llm, "invoke"):
            resp = llm.invoke(prompt)  # type: ignore
            if isinstance(resp, str):
                return resp
            # For ChatOpenAI, resp is an AIMessage
            return getattr(resp, "content", str(resp))
        # Fallback to __call__ interface
        return llm(prompt)
    except Exception as e:
        return f"[Error generating response] {e}"


def generate_summary(llm, embeddings) -> str:
    vectordb = load_or_create_vectordb(embeddings)
    # Pull a larger set of docs for summarization
    try:
        docs = vectordb.get(include=["documents", "metadatas"])  # type: ignore
        raw_texts = docs.get("documents", []) if isinstance(docs, dict) else []
        if not raw_texts:
            return "No content available to summarize. Please upload or ingest documents."
        # Concatenate a subset to respect token limits
        joined = "\n\n".join([t for t in raw_texts[:20]])[:12000]
        prompt = (
            "You are a concise summarizer. Create a clear, structured summary of the following knowledge base contents."
            " Use bullet points and short paragraphs.\n\n"
            f"Contents:\n{joined}\n\nSummary:"
        )
        return generate_response(llm, prompt)
    except Exception:
        return "No content available to summarize. Please upload or ingest documents."


def render_chat_message(role: str, text: str):
    is_user = role == "user"
    bubble_class = "user-bubble" if is_user else "ai-bubble"
    avatar = "👤" if is_user else "🤖"
    st.markdown(
        f"""
        <div class="chat-row">
            <div class="avatar">{avatar}</div>
            <div class="{bubble_class}">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    load_dotenv()
    ensure_dirs()

    st.set_page_config(page_title="AI Knowledge Assistant", page_icon="📚", layout="wide")

    # Inject CSS for mobile-friendly UI
    st.markdown(
        """
        <style>
        :root { --gap: 12px; }
        .main .block-container { padding-top: 1rem; padding-bottom: 6rem; }
        /* Chat rows */
        .chat-row { display: flex; align-items: flex-start; gap: var(--gap); margin-bottom: var(--gap); }
        .avatar { font-size: 22px; padding-top: 6px; }
        .user-bubble, .ai-bubble {
            padding: 12px 14px; border-radius: 14px; max-width: 920px; line-height: 1.5;
            box-shadow: 0 1px 2px rgba(0,0,0,0.06); word-wrap: break-word; white-space: pre-wrap;
        }
        .user-bubble { background: #e8f0fe; color: #0b57d0; }
        .ai-bubble { background: #e6f4ea; color: #0d652d; }
        /* Card */
        .card { background: #ffffff; border: 1px solid #eee; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .sources { font-size: 0.95rem; color: #444; }
        /* Sticky bottom input */
        .stChatInput, .sticky-input { position: fixed; bottom: 0; left: 0; right: 0; background: rgba(255,255,255,0.98); border-top: 1px solid #eee; padding: 10px 16px; z-index: 1000; }
        .sticky-inner { max-width: 1100px; margin: 0 auto; display: flex; gap: 8px; }
        .sticky-inner .stTextInput>div>div>input { height: 46px; font-size: 16px; }
        .sticky-inner .stButton>button { height: 46px; }
        @media (max-width: 768px) {
            .user-bubble, .ai-bubble { font-size: 0.98rem; }
            .sticky-inner .stTextInput>div>div>input { height: 42px; font-size: 15px; }
            .sticky-inner .stButton>button { height: 42px; font-size: 15px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        default_key = os.environ.get("OPENAI_API_KEY", "")
        default_gemini = os.environ.get("GEMINI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=default_key, type="password")
        gemini_key = st.text_input("Gemini API Key", value=default_gemini, type="password")
        model_choice = st.selectbox("Model", [
            "gpt-3.5-turbo",
            "gpt-4",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "HuggingFace local fallback"
        ], index=0)
        st.caption("Vector DB: ChromaDB (local). Embeddings: OpenAI/Gemini with HF fallback.")

        st.divider()
        st.subheader("Ingest Documents")
        uploaded_files = st.file_uploader("Upload PDFs or TXTs", type=["pdf", "txt"], accept_multiple_files=True)
        if uploaded_files:
            with st.spinner("Adding to knowledge base..."):
                count = add_uploaded_files_to_chroma(uploaded_files, get_embeddings(api_key, gemini_key))
            st.success(f"Added {count} chunks to Chroma.")

    # Session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dict {role, content, sources}

    st.title("📚 AI Knowledge Assistant")
    st.markdown("Ask questions about your documents. Upload files in the sidebar.")

    # Render chat history
    for m in st.session_state.messages:
        render_chat_message(m["role"], m["content"])
        if m.get("sources"):
            with st.container():
                st.markdown(f"<div class='card sources'><b>Sources</b><br>{m['sources']}</div>", unsafe_allow_html=True)

    # Sticky input area
    with st.container():
        st.markdown("<div class='sticky-input'><div class='sticky-inner'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([6, 1, 1])
        with col1:
            user_query = st.text_input("Ask anything...", key="user_query", label_visibility="collapsed")
        with col2:
            ask_clicked = st.button("Ask")
        with col3:
            summarize_clicked = st.button("Summarize")
        st.markdown("</div></div>", unsafe_allow_html=True)

    def handle_ask(q: str):
        if not q.strip():
            return
        embeddings = get_embeddings(api_key, gemini_key)
        ctx_docs = retrieve_context(q, embeddings)
        prompt = build_prompt(q, ctx_docs)
        llm = get_llm(model_choice, api_key, gemini_key)
        answer = generate_response(llm, prompt)
        sources_md = format_sources(ctx_docs)
        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_md})

    def handle_summarize():
        embeddings = get_embeddings(api_key, gemini_key)
        llm = get_llm(model_choice, api_key, gemini_key)
        summary = generate_summary(llm, embeddings)
        st.session_state.messages.append({"role": "assistant", "content": summary, "sources": None})

    if ask_clicked:
        handle_ask(user_query)
        st.experimental_rerun()

    if summarize_clicked:
        handle_summarize()
        st.experimental_rerun()


if __name__ == "__main__":
    main()


