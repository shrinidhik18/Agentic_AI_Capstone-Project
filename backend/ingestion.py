import os
import glob
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.vectorstore import VectorStoreManager

logger = logging.getLogger("DocumentIngestion")

DEFAULT_SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_documents")

def infer_doc_type(filename: str, text_content: str) -> str:
    """Infer document classification metadata based on filename and content."""
    fn_lower = filename.lower()
    content_lower = text_content[:500].lower()

    if "regulation" in fn_lower or "regulation" in content_lower or "attendance" in content_lower:
        return "regulation"
    elif "exam" in fn_lower or "hall ticket" in content_lower or "re-evaluation" in content_lower:
        return "examination"
    elif "intern" in fn_lower or "noc" in content_lower or "t&p" in content_lower:
        return "internship"
    elif "syllab" in fn_lower or "course" in content_lower or "module" in content_lower:
        return "syllabus"
    elif "faq" in fn_lower or "question" in content_lower:
        return "faq"
    return "general"


def load_file(file_path: str) -> List[Document]:
    """Load single file (PDF, TXT, MD) and return standard Document list."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    raw_text = ""

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page in reader.pages:
                raw_text += page.extract_text() or ""
        except Exception as e:
            logger.error(f"Failed to read PDF {file_path}: {e}")
            return []
    elif ext in [".txt", ".md", ".markdown"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return []
    else:
        logger.warning(f"Unsupported file format: {ext}")
        return []

    doc_type = infer_doc_type(filename, raw_text)
    return [Document(
        page_content=raw_text,
        metadata={
            "source": filename,
            "doc_type": doc_type,
            "file_path": file_path
        }
    )]


def ingest_directory(directory_path: str = DEFAULT_SAMPLE_DIR, chunk_size: int = 500, chunk_overlap: int = 80) -> int:
    """
    Ingests all documents from the specified directory into the VectorStoreManager.
    """
    if not os.path.exists(directory_path):
        logger.warning(f"Directory {directory_path} does not exist.")
        return 0

    files = glob.glob(os.path.join(directory_path, "*.*"))
    all_raw_docs = []

    for f in files:
        if f.endswith((".pdf", ".txt", ".md", ".markdown")):
            docs = load_file(f)
            all_raw_docs.extend(docs)

    if not all_raw_docs:
        logger.warning("No valid documents found for ingestion.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )

    chunked_docs = []
    for idx, doc in enumerate(all_raw_docs):
        chunks = splitter.split_documents([doc])
        for c_idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{doc.metadata['source']}_c{c_idx}"
            chunked_docs.append(chunk)

    vs = VectorStoreManager()
    vs.add_documents(chunked_docs)

    logger.info(f"Ingestion complete: {len(chunked_docs)} chunks generated from {len(all_raw_docs)} files.")
    return len(chunked_docs)
