"""
Chunk every Docling-converted policy markdown file, embed the chunks with a
local sentence-transformers model, and upsert them into a persistent local
Chroma collection. Run this after policy_process.py whenever the source
documents in data/docs/ change.
"""

import glob
import os

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DOCS_GLOB = os.path.join("data", "docs", "*.md")
VECTORSTORE_DIR = os.path.join("data", "vectorstore")
COLLECTION_NAME = "policy_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def build_index() -> int:
    md_paths = sorted(glob.glob(DOCS_GLOB))
    if not md_paths:
        raise FileNotFoundError(
            f"No markdown docs found matching {DOCS_GLOB}. Run policy_process.py first."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    documents, metadatas, ids = [], [], []
    for path in md_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.splitext(os.path.basename(path))[0]
        for i, chunk in enumerate(splitter.split_text(text)):
            documents.append(chunk)
            metadatas.append({"source": source, "chunk_index": i})
            ids.append(f"{source}-{i}")

    if not documents:
        raise ValueError("No chunks produced from the policy documents.")

    print(f"Embedding {len(documents)} chunks from {len(md_paths)} document(s)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(documents, show_progress_bar=False).tolist()

    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

    print(f"Indexed {len(documents)} chunks into '{VECTORSTORE_DIR}' (collection: {COLLECTION_NAME}).")
    return len(documents)


if __name__ == "__main__":
    build_index()
