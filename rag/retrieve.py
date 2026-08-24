"""
Embedding similarity search over the indexed policy chunks, followed by a
cross-encoder rerank pass. This replaces the old substring/keyword match —
retrieval is now a real (if small-scale) RAG pipeline: chunk -> embed ->
vector search -> rerank.
"""

from functools import lru_cache

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.ingest import COLLECTION_NAME, EMBEDDING_MODEL_NAME, VECTORSTORE_DIR

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVE = 8
TOP_N_RERANKED = 3


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL_NAME)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve_policy(query: str, top_k: int = TOP_K_RETRIEVE, top_n: int = TOP_N_RERANKED) -> list[str]:
    """
    Returns the top_n reranked policy chunk texts for `query`, most relevant
    first. Returns [] if the vector index hasn't been built yet
    (rag/ingest.py hasn't been run) rather than raising.
    """
    try:
        collection = _collection()
    except Exception:
        return []

    query_embedding = _embedder().encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    candidates = results.get("documents", [[]])[0]
    if not candidates:
        return []

    pairs = [(query, doc) for doc in candidates]
    scores = _reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]


def retrieve_policy_debug(query: str, top_k: int = TOP_K_RETRIEVE, top_n: int = TOP_N_RERANKED) -> list[dict]:
    """
    Like retrieve_policy, but returns per-chunk detail (source document,
    pre-rerank vector rank, post-rerank rank, rerank score) instead of just
    text. Used by eval/ to measure retrieval quality, not by the live agent.
    """
    collection = _collection()
    query_embedding = _embedder().encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return []

    pairs = [(query, doc) for doc in documents]
    scores = _reranker().predict(pairs)

    pre_rank = {i: i for i in range(len(documents))}
    post_order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
    post_rank = {i: rank for rank, i in enumerate(post_order)}

    debug = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        debug.append(
            {
                "text": doc,
                "source": meta.get("source"),
                "pre_rerank_rank": pre_rank[i],
                "post_rerank_rank": post_rank[i],
                "rerank_score": float(scores[i]),
                "in_top_n": post_rank[i] < top_n,
            }
        )
    return sorted(debug, key=lambda d: d["post_rerank_rank"])
