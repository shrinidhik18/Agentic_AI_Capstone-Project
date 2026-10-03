import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from backend.llm_config import get_embeddings

logger = logging.getLogger("VectorStoreManager")

DB_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_db")

class VectorStoreManager:
    def __init__(self, persist_directory: str = DB_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        self.vector_store = None
        self._initialize_or_load()

    def _initialize_or_load(self):
        """Load vector store if exists on disk, else create empty container."""
        if os.path.exists(self.persist_directory) and os.path.exists(os.path.join(self.persist_directory, "index.faiss")):
            try:
                from langchain_community.vectorstores import FAISS
                self.vector_store = FAISS.load_local(
                    self.persist_directory, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded existing FAISS vector store from disk.")
            except Exception as e:
                logger.warning(f"Could not load FAISS store: {e}. Will re-index on demand.")
                self.vector_store = None

    def add_documents(self, documents: List[Document]):
        """Index documents into FAISS vector store and persist to disk."""
        if not documents:
            return
        
        from langchain_community.vectorstores import FAISS
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store.save_local(self.persist_directory)
        logger.info(f"Persisted {len(documents)} document chunks to FAISS vector store at {self.persist_directory}.")

    def similarity_search_with_score(
        self, query: str, top_k: int = 4, relevance_threshold: float = 0.2
    ) -> List[Tuple[Document, float]]:
        """
        Performs similarity search returning top-k matching documents and normalized relevance scores.
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty! Please ingest documents first.")
            return []

        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            # FAISS distance conversion to 0-1 similarity score
            processed = []
            for doc, distance in results:
                # FAISS uses L2 distance (lower is better) or inner product.
                # Convert distance into similarity score roughly [0, 1]
                similarity = 1.0 / (1.0 + float(distance))
                if similarity >= relevance_threshold:
                    processed.append((doc, round(similarity, 4)))
            
            # Sort descending by score
            processed.sort(key=lambda x: x[1], reverse=True)
            return processed
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def clear(self):
        """Clears the vector store."""
        self.vector_store = None
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            logger.info("Cleared vector store persist directory.")
