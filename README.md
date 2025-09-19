# AI Knowledge Assistant (Streamlit + ChromaDB)

Local, mobile-friendly RAG app using free tools by default. OpenAI is optional.

## Quickstart

1) Create and activate a virtual environment (recommended).

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # PowerShell on Windows
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) (Optional) Set OPENAI_API_KEY for best quality:

- PowerShell:
```bash
$env:OPENAI_API_KEY="sk-..."
```

Without a key, the app uses free Hugging Face embeddings and a small local text generation model.

4) Ingest your documents:

- Place PDFs/TXT into ./docs
- Run:
```bash
python ingest.py
```

5) Start the app:

```bash
streamlit run app.py
```

Then open the provided local URL in your browser (works on mobile too).

## Notes
- Vector DB: ChromaDB at ./chroma_db
- Embeddings: OpenAI text-embedding-ada-002 when key present; otherwise sentence-transformers/all-MiniLM-L6-v2
- LLM: OpenAI gpt-3.5-turbo/gpt-4 when key present; local Transformers fallback otherwise
- Uploading files in the sidebar also ingests them into Chroma

## Troubleshooting
- If Chroma complains about versions, clean it with:
```bash
Remove-Item -Recurse -Force chroma_db
```
- For large PDFs, ingestion may take time. Watch console logs.
- If Transformers downloads are slow, it's a one-time model download.
# Inquiro
