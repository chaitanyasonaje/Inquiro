import os
import glob
from typing import List

from dotenv import load_dotenv

# LangChain core & community
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma

# Embeddings
try:
    # Prefer the official LangChain OpenAI integration
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


DB_DIR = os.path.join(".", "chroma_db")
DOCS_DIR = os.path.join(".", "docs")


def ensure_dirs() -> None:
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)


def get_embeddings():
    """Return embeddings with priority: OpenAI -> Gemini -> HF fallback."""
    load_dotenv()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if openai_key and HAS_LANGCHAIN_OPENAI:
        # OpenAI Embeddings default per spec: text-embedding-ada-002
        # Newer API uses text-embedding-3 variants but we honor the requested default.
        return OpenAIEmbeddings(model="text-embedding-ada-002")

    if gemini_key and HAS_GOOGLE_EMBED:
        # Use Gemini embedding model
        # Typical model: models/embedding-001
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)

    # Fallback: sentence-transformers (free)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_documents_from_dir(path: str) -> List:
    files = []
    files.extend(glob.glob(os.path.join(path, "**", "*.pdf"), recursive=True))
    files.extend(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))

    documents = []
    for f in files:
        try:
            if f.lower().endswith(".pdf"):
                loader = PyPDFLoader(f)
            else:
                loader = TextLoader(f, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")
    return documents


def split_documents(docs: List):
    # Token-based splitting using tiktoken via LangChain
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)


def build_or_update_chroma(splits: List):
    embeddings = get_embeddings()
    # Create or load collection
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="documents",
    )
    if splits:
        vectordb.add_documents(splits)
        vectordb.persist()
    return vectordb


def main():
    ensure_dirs()
    print(f"[INFO] Loading documents from {DOCS_DIR} ...")
    docs = load_documents_from_dir(DOCS_DIR)
    if not docs:
        print("[INFO] No documents found in ./docs. Add PDFs or TXTs and rerun.")
        return

    print(f"[INFO] Loaded {len(docs)} raw documents. Splitting into chunks ...")
    splits = split_documents(docs)
    print(f"[INFO] Produced {len(splits)} chunks. Generating embeddings and updating Chroma ...")
    build_or_update_chroma(splits)
    print(f"[SUCCESS] Ingestion complete. Chroma DB at: {DB_DIR}")


if __name__ == "__main__":
    main()


