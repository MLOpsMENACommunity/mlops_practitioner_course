import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Method 1: Mean cosine similarity drift ────────────────────
def embedding_cosine_drift(ref_texts, curr_texts, threshold=0.85):
    """
    Encode both sets and compare mean embeddings.
    Cosine < threshold signals the semantic space has shifted.
    """
    ref_emb  = encoder.encode(ref_texts,  normalize_embeddings=True)
    curr_emb = encoder.encode(curr_texts, normalize_embeddings=True)
    # Mean embedding per period
    ref_mean  = ref_emb.mean(axis=0, keepdims=True)
    curr_mean = curr_emb.mean(axis=0, keepdims=True)
    drift_score = float(cosine_similarity(ref_mean, curr_mean)[0][0])
    print(f"Embedding cosine similarity: {drift_score:.4f}")
    if drift_score < threshold:
        print("⚠ Embedding drift detected — query space has shifted")
    return drift_score

# ── Method 2: Maximum Mean Discrepancy (MMD) ──────────────────
def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma=1.0) -> float:
    """MMD with RBF kernel. MMD ≈ 0 means same distribution."""
    from sklearn.metrics.pairwise import rbf_kernel
    XX = rbf_kernel(X, X, gamma).mean()
    YY = rbf_kernel(Y, Y, gamma).mean()
    XY = rbf_kernel(X, Y, gamma).mean()
    return float(XX + YY - 2 * XY)

# ── Use case: RAG query monitoring ────────────────────────────
# Track weekly: if users start asking a different TYPE of question,
# the retriever's embedding space may no longer match their intent.
ref_queries  = load_queries("2024-01")   # reference month
curr_queries = load_queries("2024-06")   # current month
mmd = mmd_rbf(encoder.encode(ref_queries), encoder.encode(curr_queries))
print(f"MMD score: {mmd:.6f}")  # > 0.01 warrants investigation
