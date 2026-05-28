import os
import json
from dotenv import load_dotenv
from google import genai
import time
from youtube_scraper.search import semantic_search


load_dotenv()


def load_video_info(output_file):
    """Load video metadata từ file output json."""
    if not os.path.exists(output_file):
        return None

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("videos", [])
    if not videos:
        return None

    return videos[0]


def build_context_from_chunks(chunks):
    """
    Ghép các chunk đã retrieve thành context đưa vào LLM.
    Mỗi chunk có timestamp để LLM trích nguồn.
    """
    context_parts = []

    for chunk in chunks:
        start_time = chunk.get("start_time", "")
        end_time = chunk.get("end_time", "")
        text = chunk.get("text", "")
        source = chunk.get("url_with_timestamp", "")

        part = f"""
[Đoạn: {start_time} - {end_time}]
{text}
Nguồn: {source}
""".strip()

        context_parts.append(part)

    return "\n\n---\n\n".join(context_parts)


def answer_with_llm(question, output_file, index_file, rag_file, top_k=3):
    """
    Trả lời tự nhiên bằng Gemini dựa trên các chunk retrieve được.
    """
    video = load_video_info(output_file)

    if not video:
        return "Không tìm thấy thông tin video. Bạn hãy kiểm tra lại file output."

    chunks = semantic_search(
        question=question,
        index_file=index_file,
        rag_file=rag_file,
        top_k=top_k
    )

    if not chunks:
        return "Mình không tìm thấy đoạn transcript phù hợp để trả lời câu hỏi này."

    context = build_context_from_chunks(chunks)

    title = video.get("title", "")
    channel = video.get("channel_title", "")
    description = video.get("description", "")
    short_description = description[:1200] if description else ""

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        return "Thiếu GEMINI_API_KEY trong file .env."

    client = genai.Client(api_key=api_key)

    prompt = f"""
Bạn là chatbot hỏi đáp dựa trên transcript video YouTube.

THÔNG TIN VIDEO:
- Tiêu đề: {title}
- Kênh: {channel}
- Mô tả ngắn: {short_description}

CONTEXT TRÍCH TỪ TRANSCRIPT:
{context}

CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

YÊU CẦU TRẢ LỜI:
1. Trả lời tự nhiên, dễ hiểu bằng tiếng Việt.
2. Chỉ dựa trên CONTEXT và thông tin video được cung cấp.
3. Không bịa thêm thông tin ngoài video.
4. Nếu context không đủ, hãy nói rõ: "Video không đề cập rõ nội dung này."
5. Cuối câu trả lời luôn có mục "Nguồn tham chiếu" liệt kê timestamp đã dùng.
6. Không cần nhắc lại toàn bộ transcript.
"""

    last_error = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            return response.text

        except Exception as e:
            last_error = e
            error_text = str(e)

            # Retry khi Gemini quá tải hoặc lỗi tạm thời
            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "overloaded" in error_text.lower()
                or "high demand" in error_text.lower()
                or "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):
                wait_time = 2 + attempt * 3

                print(f"Gemini đang quá tải. Thử lại lần {attempt + 1}/{max_retries} sau {wait_time}s...")
                time.sleep(wait_time)
                continue

            # Nếu là lỗi khác thì không retry nữa
            return f"Lỗi khi gọi Gemini API: {e}"


    # Nếu retry hết 3 lần vẫn lỗi, fallback sang kết quả RAG
    fallback_sources = []

    for chunk in chunks:
        fallback_sources.append(
            f"- {chunk.get('start_time', '')} - {chunk.get('end_time', '')}: {chunk.get('url_with_timestamp', '')}"
        )

    return (
        "Gemini đang quá tải nên chưa sinh được câu trả lời tự nhiên.\n\n"
        "Tuy nhiên hệ thống đã tìm được các đoạn transcript liên quan nhất:\n\n"
        f"{context[:1800]}...\n\n"
        "Nguồn tham chiếu:\n" + "\n".join(fallback_sources)
    )