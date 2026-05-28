# YouTube RAG Studio

**YouTube RAG Studio** là hệ thống hỏi đáp thông minh trên nội dung video YouTube, ứng dụng quy trình **Retrieval-Augmented Generation (RAG)** để crawl transcript, chia nhỏ nội dung, tạo vector index và trả lời câu hỏi dựa trên ngữ cảnh video.

## Tính năng chính

- Crawl metadata video YouTube bằng YouTube Data API.
- Tải phụ đề/transcript bằng `yt-dlp`.
- Làm sạch transcript và chia thành các chunk phục vụ RAG.
- Tạo embedding tiếng Việt bằng Sentence Transformers.
- Lưu vector index bằng FAISS.
- Hỏi đáp nội dung video bằng Gemini.
- Kiểm chứng truy xuất với top chunks, timestamp, similarity score và rerank score.
- Web UI local để crawl video, chọn dataset và hỏi đáp.
- Tùy chọn tự transcribe audio bằng `faster-whisper` khi video không có phụ đề.

## Công nghệ sử dụng

- Python
- YouTube Data API
- yt-dlp
- Sentence Transformers
- FAISS
- Google Gemini API
- faster-whisper
- HTML/CSS/JavaScript
- Local HTTP server bằng `http.server`

## Kiến trúc tổng quan

```text
YouTube URL
   |
   v
Fetch metadata + transcript
   |
   v
Clean transcript
   |
   v
Chunk transcript
   |
   v
Embedding model
   |
   v
FAISS vector index
   |
   v
Semantic search + reranking
   |
   v
LLM answer generation
```

## Cấu trúc thư mục

```text
ytb-rag-llm/
├── web_app.py
├── requirements.txt
├── youtube_scraper/
│   ├── main.py
│   ├── metadata.py
│   ├── transcripts.py
│   ├── utils.py
│   ├── knowledge_base.py
│   ├── search.py
│   └── chatbot.py
├── local/
│   └── <video-dataset>/
│       ├── <name>.json
│       ├── <name>_rag.jsonl
│       ├── <name>_embeddings.npy
│       └── <name>_vector_index.faiss
└── .env
```

## Cài đặt

### 1. Clone project

```bash
git clone https://github.com/<your-username>/ytb-rag-llm.git
cd ytb-rag-llm
```

### 2. Tạo virtual environment

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Cài dependencies

```powershell
pip install -r requirements.txt
```

Nếu dùng tính năng tự transcribe audio khi video không có phụ đề, cần cài thêm `ffmpeg` trên máy.

## Cấu hình môi trường

Tạo file `.env` tại thư mục gốc:

```env
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Nếu dùng `faster-whisper`, có thể cấu hình thêm:

```env
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

## Chạy web app

```powershell
python web_app.py
```

Mặc định app chạy tại:

```text
http://127.0.0.1:8000
```

Nếu muốn đổi port:

```powershell
python web_app.py --port 8002
```

## Cách sử dụng web app

1. Dán link video YouTube vào ô nhập.
2. Bấm **Go** để xử lý video.
3. Hệ thống sẽ:
   - lấy metadata,
   - tải transcript,
   - tạo RAG chunks,
   - tạo embeddings,
   - build FAISS index.
4. Chọn video trong thư viện.
5. Nhập câu hỏi và bấm **Ask**.
6. Mở mục **Kiểm chứng truy xuất** để xem các chunk liên quan, điểm similarity/rerank và timestamp.

## Chạy bằng CLI

Crawl một video và build knowledge base:

```powershell
python -m youtube_scraper.main --video "https://www.youtube.com/watch?v=VIDEO_ID" --output local\demo\demo.json --rag --knowledge-base --delay 15
```

Hỏi đáp bằng semantic search:

```powershell
python -m youtube_scraper.main --ask "Video này nói về gì?" --output local\demo\demo.json
```

Chế độ chat:

```powershell
python -m youtube_scraper.main --chat --output local\demo\demo.json
```

Fallback transcribe audio nếu không có phụ đề:

```powershell
python -m youtube_scraper.main --video "https://www.youtube.com/watch?v=VIDEO_ID" --output local\demo\demo.json --rag --knowledge-base --transcribe-missing
```

## Các file dữ liệu sinh ra

Với mỗi video, hệ thống tạo:

```text
<name>.json                    Metadata + transcript
<name>_rag.jsonl               Transcript chunks dùng cho RAG
<name>_embeddings.npy          Embedding vectors
<name>_vector_index.faiss      FAISS vector index
```

## Câu hỏi demo gợi ý

```text
Video này nói về nội dung gì?
```

```text
Các bước chính trong video là gì?
```

```text
Trong file dữ liệu có những cột nào dùng để phân tích?
```

```text
Tìm đoạn video nói về việc import dữ liệu vào Power BI.
```

```text
Video này phù hợp với người học kỹ năng nào?
```



