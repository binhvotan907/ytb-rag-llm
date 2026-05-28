import os
import json
import re


MODEL_NAME = "keepitreal/vietnamese-sbert"
_EMBED_MODEL = None

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANKER_MODEL = None
_RAG_CACHE = {}
_INDEX_CACHE = {}

def get_embedding_model():
    """Load embedding model once."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model ({MODEL_NAME})...")
        _EMBED_MODEL = SentenceTransformer(MODEL_NAME)
    return _EMBED_MODEL

def get_reranker_model():
    """Load reranker model once."""
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker model ({RERANKER_MODEL_NAME})...")
        _RERANKER_MODEL = CrossEncoder(RERANKER_MODEL_NAME)
    return _RERANKER_MODEL


def load_rag_chunks(rag_file):
    """Load and cache RAG chunks for the current web process."""
    cache_key = (rag_file, os.path.getmtime(rag_file))
    if cache_key not in _RAG_CACHE:
        chunks = []
        with open(rag_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        _RAG_CACHE.clear()
        _RAG_CACHE[cache_key] = chunks
    return _RAG_CACHE[cache_key]


def load_faiss_index(index_file):
    """Load and cache FAISS index for the current web process."""
    import faiss

    cache_key = (index_file, os.path.getmtime(index_file))
    if cache_key not in _INDEX_CACHE:
        _INDEX_CACHE.clear()
        _INDEX_CACHE[cache_key] = faiss.read_index(index_file)
    return _INDEX_CACHE[cache_key]


def semantic_search(question, index_file, rag_file, top_k=3):
    """
    Search chunks bằng FAISS trước, sau đó dùng Cross-Encoder reranker
    để chọn chunk liên quan nhất.
    """
    try:
        import faiss
    except ImportError:
        print("\n ERROR: Required ML libraries for semantic search are missing.")
        print("   Install them with: pip install sentence-transformers faiss-cpu numpy")
        return []

    if not os.path.exists(index_file):
        print(f"\n ERROR: FAISS index not found at {index_file}")
        print("   Run with --knowledge-base first to build the index.")
        return []

    if not os.path.exists(rag_file):
        print(f"\n ERROR: RAG dataset not found at {rag_file}")
        return []

    chunks = load_rag_chunks(rag_file)

    if not chunks:
        print("   No chunks found in RAG dataset.")
        return []

    index = load_faiss_index(index_file)

    # Bước 1: FAISS retrieve rộng hơn
    model = get_embedding_model()
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    candidate_k = min(20, len(chunks))
    scores, indices = index.search(query_embedding, candidate_k)

    candidates = []

    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = dict(chunks[idx])
        chunk["score"] = float(scores[0][rank])
        candidates.append(chunk)

    if not candidates:
        return []

    # Bước 2: Rerank bằng Cross-Encoder
    reranker = get_reranker_model()

    pairs = [
        [question, chunk.get("text", "")]
        for chunk in candidates
    ]

    rerank_scores = reranker.predict(pairs)

    for chunk, rerank_score in zip(candidates, rerank_scores):
        chunk["rerank_score"] = float(rerank_score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    final_results = candidates[:top_k]

    for rank, chunk in enumerate(final_results, start=1):
        chunk["rank"] = rank

    return final_results


def print_search_results(question, results, preview_chars=280):
    """Pretty-print semantic search results to stdout."""
    print(f"\n{'=' * 70}")
    print(" Question:")
    print(f"   {question}")
    print(f"{'=' * 70}")

    if not results:
        print("\n   No results found. Build a knowledge base first with --knowledge-base.")
        return

    print(f"\n Top {len(results)} Results:\n")

    for result in results:
        title = result.get("title", "Unknown")
        channel = result.get("channel", "Unknown")
        text = result.get("text", "")
        score = result.get("score", 0.0)
        rank = result.get("rank", 0)

        preview = text[:preview_chars].strip()
        if len(text) > preview_chars:
            preview += "..."

        start_time = result.get("start_time", "")
        end_time = result.get("end_time", "")
        url_with_timestamp = result.get("url_with_timestamp", "")

        print(f"[{rank}] {channel} – {title}")
        print(f"    Similarity: {score:.4f}")

        if "rerank_score" in result:
            print(f"    Rerank    : {result['rerank_score']:.4f}")

        if start_time and end_time:
            print(f"    Time      : {start_time} - {end_time}")

        if url_with_timestamp:
            print(f"    Source    : {url_with_timestamp}")

        print(f"    Preview   : {preview}\n")

def is_overview_question(question: str) -> bool:
    """
    Kiểm tra câu hỏi có phải dạng hỏi tổng quan video không.
    """
    q = question.lower().strip()

    overview_patterns = [
        "video này nói về",
        "nội dung video",
        "video nói gì",
        "video này là gì",
        "tóm tắt video",
        "tóm tắt nội dung",
        "ý chính của video",
        "chủ đề của video",
        "video này hướng dẫn gì",
        "video này trình bày gì"
    ]

    return any(pattern in q for pattern in overview_patterns)


def load_video_info(output_file):
    """
    Load thông tin video từ file JSON chính, ví dụ local/test.json.
    """
    if not os.path.exists(output_file):
        return None

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("videos", [])
    if not videos:
        return None

    return videos[0]


def _extract_timestamp_topics(description):
    """Extract timestamp labels from a YouTube description."""
    topics = []
    for line in description.splitlines():
        line = line.strip()
        match = re.match(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)$", line)
        if match:
            topics.append(f"{match.group(1)} - {match.group(2).strip()}")
    return topics


def _first_sentences(text, max_sentences=5):
    """Return a few readable sentences from text."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s.strip() for s in sentences if s.strip()][:max_sentences]


def print_video_overview(output_file):
    """
    Trả lời câu hỏi tổng quan dựa vào metadata + description.
    """
    video = load_video_info(output_file)

    if not video:
        print("Không tìm thấy thông tin video.")
        return

    title = video.get("title", "")
    channel = video.get("channel_title", "")
    description = video.get("description", "")
    duration = video.get("duration", "")
    url = video.get("url", "")
    transcript = video.get("transcript", "")

    print("\n" + "=" * 70)
    print(" TỔNG QUAN VIDEO")
    print("=" * 70)

    print(f"\n🎬 Tiêu đề: {title}")
    print(f" Kênh: {channel}")
    print(f" Thời lượng: {duration}")
    print(f" Link: {url}")

    # Lấy phần mô tả đầu tiên, tránh in quá dài
    short_description = description.split("---------------------------------")[0].strip()

    if short_description:
        print("\n Mô tả từ video:")
        print(short_description[:1000])

    topics = _extract_timestamp_topics(description)
    intro_sentences = _first_sentences(transcript, max_sentences=5)

    print("\n Nội dung chính từ video:")
    if topics:
        for topic in topics[:8]:
            print(f"- {topic}")
    elif intro_sentences:
        for sentence in intro_sentences:
            print(f"- {sentence}")
    else:
        print("- Chưa có transcript để suy ra nội dung chi tiết.")

    if transcript:
        preview = transcript[:800].strip()
        print("\n Đoạn mở đầu transcript:")
        print(preview + "...")        
