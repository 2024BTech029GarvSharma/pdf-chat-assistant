"""
vector_store.py
----------------
The heart of the RAG system: CHUNKING, EMBEDDING, and STORAGE/SEARCH.

This version chunks text PER PAGE (instead of chunking the whole document
as one long string). That means every chunk knows exactly which page it
came from, which lets us show the user "page 3" instead of just a
filename when explaining where an answer came from.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def chunk_text(text, chunk_size=300, overlap=40):
    """
    Splits text into overlapping word chunks. Chunk size is smaller here
    (300 vs the original 500) because we now chunk PER PAGE, and most
    pages aren't huge - smaller chunks keep page-level chunks focused.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


class VectorStore:
    """
    Wraps a FAISS index + metadata about each vector (source file, page
    number, and the actual chunk text).
    """

    def __init__(self):
        self.index = None
        self.chunk_metadata = []  # list of {"text", "source", "page"}
        self.embedding_dim = None

    def build_from_documents(self, documents: dict):
        """
        Parameters:
            documents: dict of { "filename.pdf": [{"page": 1, "text": "..."}, ...] }
                       (this is exactly what pdf_utils.py now produces)
        """
        model = get_embedding_model()
        all_chunks = []
        self.chunk_metadata = []

        for filename, pages in documents.items():
            for page_entry in pages:
                page_number = page_entry["page"]
                page_chunks = chunk_text(page_entry["text"])
                for chunk in page_chunks:
                    all_chunks.append(chunk)
                    self.chunk_metadata.append(
                        {"text": chunk, "source": filename, "page": page_number}
                    )

        if not all_chunks:
            self.index = None
            return

        embeddings = model.encode(all_chunks, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")
        self.embedding_dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 4):
        """
        Finds the top_k most relevant chunks for a question.

        Returns:
            list of {"text", "source", "page", "distance"}
        """
        if self.index is None:
            return []

        model = get_embedding_model()
        query_vector = model.encode([query], normalize_embeddings=True).astype("float32")

        similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            result = dict(self.chunk_metadata[idx])
            result["similarity"] = float(similarities[0][rank])
            results.append(result)

        return results

    def get_all_embeddings(self):
        """
        Pulls the raw embedding vectors back out of the FAISS index.

        FAISS stores vectors internally but doesn't expose them by default -
        this uses reconstruct_n (available on IndexFlatL2) to retrieve them.
        Useful for demos/visualization - showing what an embedding actually
        looks like, not just using it invisibly for search.

        Returns:
            numpy array of shape (num_chunks, embedding_dim), or None if
            no documents have been processed yet.
        """
        if self.index is None or self.index.ntotal == 0:
            return None
        return self.index.reconstruct_n(0, self.index.ntotal)
