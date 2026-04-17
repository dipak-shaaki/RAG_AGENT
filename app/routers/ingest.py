import os
import shutil
from fastapi import APIRouter, UploadFile, File
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

router = APIRouter()

DOCS_DIR = "docs/"
CHROMA_PATH = "chroma_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_and_save_vectorstore():
    """
    Loads all .txt and .pdf files from docs/,
    chunks them, embeds them, saves to Chroma on disk.
    Chroma handles persistence automatically — no manual save needed.
    """
    # Delete existing index to ensure fresh start
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"[Ingest] Removed existing Chroma index at {CHROMA_PATH}")

    # Load TXT files
    txt_loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")
    )

    # Load PDF files
    pdf_loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    # Combine both
    documents = txt_loader.load() + pdf_loader.load()

    if not documents:
        raise ValueError("No documents found in docs/ folder")

    print(f"[Ingest] Loaded {len(documents)} documents")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    print(f"[Ingest] Created {len(chunks)} chunks")

    # Embed
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Store in Chroma — persists automatically to chroma_index/
    # if collection already exists, it gets replaced with new docs
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="documents",
        persist_directory=CHROMA_PATH
    )

    print(f"[Ingest] Chroma index saved to {CHROMA_PATH}/")

    return len(chunks)


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    POST /ingest
    Upload a .txt or .pdf file → chunk → embed → save to Chroma
    """
    # Step 1 — save uploaded file to docs/
    os.makedirs(DOCS_DIR, exist_ok=True)
    file_path = os.path.join(DOCS_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"[Ingest] Saved file: {file_path}")

    # Step 2 — rebuild vectorstore with all docs

    total_chunks = build_and_save_vectorstore()

    return {
        "message": f"'{file.filename}' ingested successfully",
        "total_chunks": total_chunks,
        "index_saved_to": CHROMA_PATH
    }
