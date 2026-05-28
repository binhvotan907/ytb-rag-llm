import os
import json


MODEL_NAME = "keepitreal/vietnamese-sbert"
_EMBED_MODEL = None


def get_embedding_model():
    """Load embedding model once."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"   Loading embedding model ({MODEL_NAME})...")
        _EMBED_MODEL = SentenceTransformer(MODEL_NAME)
    return _EMBED_MODEL


def build_knowledge_base(output_file):
    """Build a FAISS vector index from a generated RAG dataset.

    Args:
        output_file (str): The path to the main output file.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        import faiss
        import numpy as np
    except ImportError:
        print("\n ERROR: Required ML libraries for knowledge base are missing.")
        print("   Please install them first:")
        print("   pip install sentence-transformers faiss-cpu numpy")
        return False

    base_name = os.path.splitext(output_file)[0]
    rag_file = f"{base_name}_rag.jsonl"

    if not os.path.exists(rag_file):
        print(f"\n ERROR: RAG dataset not found at {rag_file}. Ensure --rag mode ran correctly.")
        return False

    embeddings_file = f"{base_name}_embeddings.npy"
    index_file = f"{base_name}_vector_index.faiss"

    print(f"\n Building AI Knowledge Base...")

    chunks = []
    texts = []

    with open(rag_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            text = chunk.get("text", "").strip()
            if text:
                chunks.append(chunk)
                texts.append(text)

    if not texts:
        print("   No transcript chunks found to embed.")
        return False

    model = get_embedding_model()

    print(f"   Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    # Save raw normalized embeddings
    np.save(embeddings_file, embeddings)
    print(f"    Saved raw embeddings to: {embeddings_file}")

    print("   Building FAISS vector index...")

    # Since embeddings are normalized, Inner Product ~= cosine similarity
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    faiss.write_index(index, index_file)
    print(f"    Saved FAISS index to: {index_file}")

    return True