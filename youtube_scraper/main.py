import os
import sys
import argparse
import subprocess
from googleapiclient.discovery import build
import googleapiclient.errors

from youtube_scraper.utils import load_existing_progress, save_progress, clean_youtube_input, export_rag_jsonl
from youtube_scraper.metadata import get_all_video_ids, get_all_playlist_video_ids, get_video_details, resolve_channel_handle
from youtube_scraper.transcripts import add_transcripts
from youtube_scraper.knowledge_base import build_knowledge_base
from youtube_scraper.search import (
    semantic_search,
    print_search_results,
    is_overview_question,
    print_video_overview
)

from dotenv import load_dotenv
load_dotenv()
from youtube_scraper.chatbot import answer_with_llm

def main():
    parser = argparse.ArgumentParser(description="YouTube Scraper: Downloader for Video Metadata & Transcripts.")
    
    # Scraping targets (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--video", type=str, help="Scrape a single YouTube video ID or URL")
    group.add_argument("--playlist", type=str, help="Scrape all videos in a YouTube Playlist ID or URL")
    group.add_argument("--channel", type=str, help="Scrape all videos in a YouTube Channel ID, URL, or @handle")
    
    # Query mode (independent of scraping)
    parser.add_argument("--ask", type=str, help="Ask a question against an existing knowledge base (skips scraping)")
    parser.add_argument("--chat", action="store_true", help="Chat naturally with an existing video knowledge base using LLM")
    # Optional Arguments
    parser.add_argument("--output", type=str, default="scraped_transcripts.json", 
                        help="Output file name (default: scraped_transcripts.json)")
    parser.add_argument("--format", type=str, choices=["json", "jsonl", "csv", "parquet"], default="json",
                        help="Dataset export format (json, jsonl, csv, parquet). Default: json")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel thread workers for transcript downloading (default: 1)")
    parser.add_argument("--rag", action="store_true",
                       help="Generate an additional chunked JSONL dataset for RAG.")
    parser.add_argument("--knowledge-base", action="store_true",
                        help="Generate embeddings & FAISS index from the RAG dataset. (Requires ML libs)")
    parser.add_argument("--langs", type=str, default="vi-orig,vi,vi-VN", 
                        help="Ngôn ngữ phụ đề (ưu tiên vi, fallback en)")
    parser.add_argument("--delay", type=int, default=4, 
                        help="Minimum delay in seconds between downloads (default: 4)")
    parser.add_argument("--cookies", type=str, default="cookies.txt", 
                        help="Path to cookies file if available (default: cookies.txt)")
    parser.add_argument("--transcribe-missing", action="store_true",
                        help="Use local faster-whisper audio transcription when subtitles are unavailable.")

    args = parser.parse_args()

    # ─── CHATBOT MODE ─────────────────────────────────────────
    if args.chat:
        output_file = args.output
        base_name = os.path.splitext(output_file)[0]
        index_file = f"{base_name}_vector_index.faiss"
        rag_file = f"{base_name}_rag.jsonl"

        print("\n CHATBOT VIDEO RAG")
        print("Gõ câu hỏi để hỏi về video.")
        print("Gõ 'exit' để thoát.\n")

        while True:
            question = input("Bạn hỏi: ").strip()

            if question.lower() in ["exit", "quit", "thoát"]:
                print("Đã thoát chatbot.")
                break

            if not question:
                continue

            print("\nBot đang trả lời...\n")

            if is_overview_question(question):
                print_video_overview(output_file)
                print("\n" + "=" * 70 + "\n")
                continue

            answer = answer_with_llm(
                question=question,
                output_file=output_file,
                index_file=index_file,
                rag_file=rag_file,
                top_k=3
            )

            print(answer)
            print("\n" + "=" * 70 + "\n")

        return
# ─── QUERY MODE ───────────────────────────────────────────
    if args.ask:
        output_file = args.output
        base_name = os.path.splitext(output_file)[0]
        index_file = f"{base_name}_vector_index.faiss"
        rag_file = f"{base_name}_rag.jsonl"

        # Nếu là câu hỏi tổng quan thì không dùng semantic search
        # vì semantic search dễ trả về các chunk rời rạc, không đúng ý tổng quát.
        if is_overview_question(args.ask):
            print_video_overview(output_file)
            return

        # Nếu là câu hỏi cụ thể thì mới search trong các chunk
        results = semantic_search(args.ask, index_file, rag_file, top_k=3)
        print_search_results(args.ask, results, preview_chars=280)
        return
    
    # ─── SCRAPING MODE ────────────────────────────────────────
    if not (args.video or args.playlist or args.channel):
        parser.error("one of --video, --playlist, --channel, --ask, or --chat is required")

    print("=" * 60)
    print("CHƯƠNG TRÌNH CRAWL YOUTUBE")
    print("=" * 60)

    # 1. API Key Validation
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("\n ERROR: YOUTUBE_API_KEY environment variable is missing.")
        print("   Please set it before running the script.")
        print("   Windows: set YOUTUBE_API_KEY=your_key_here")
        print("   Mac/Linux: export YOUTUBE_API_KEY=your_key_here\n")
        sys.exit(1)

    # 2. Dependency Check & Warning
    if not os.path.exists(args.cookies):
        print(f"\n WARNING: {args.cookies} not found. Proceeding without cookies (may fail for age-restricted videos).")

    # 3. Setup Config
    output_file = args.output
    target_languages = [l.strip() for l in args.langs.split(",")]
    export_format = args.format
    rag_mode = args.rag or args.knowledge_base
    workers = args.workers

    # Clean the input target safely
    if args.video:
        _, args.video = clean_youtube_input(args.video)
    if args.playlist:
        _, args.playlist = clean_youtube_input(args.playlist)
    if args.channel:
        _, args.channel = clean_youtube_input(args.channel)
        
    source_target = args.video or args.playlist or args.channel

    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n ERROR: yt-dlp not found. Please install it with: pip install yt-dlp")
        sys.exit(1)

    # 4. Resume from existing progress
    existing_videos = load_existing_progress(output_file, export_format=export_format)
    existing_video_ids = {v["id"] for v in existing_videos}

    # 5. Fetch YouTube Metadata
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        new_video_ids = []

        if args.video:
            print(f"Đang lấy thông tin video: {args.video}")
            new_video_ids = [args.video]
        elif args.playlist:
            new_video_ids = get_all_playlist_video_ids(youtube, args.playlist)
        elif args.channel:
            # Check if it needs handle resolution first
            if args.channel.startswith("@"):
                resolved_id = resolve_channel_handle(youtube, args.channel)
                if not resolved_id:
                    sys.exit(1)
                args.channel = resolved_id
                
            new_video_ids = get_all_video_ids(youtube, args.channel)

        # Identify videos we do not quickly have metadata for yet
        ids_to_fetch = [vid for vid in new_video_ids if vid not in existing_video_ids]
        
        if ids_to_fetch:
            print(f"\nTìm thấy {len(ids_to_fetch)} video mới...")
            new_videos = get_video_details(youtube, ids_to_fetch)
            existing_videos.extend(new_videos)

        # Save metadata instantly
        videos = existing_videos
        save_progress(output_file, source_target, videos, export_format=export_format)

    except googleapiclient.errors.HttpError as e:
        print(f"\n YouTube API Error: {e.reason}")
        sys.exit(1)

    # 6. Fetch Transcripts for current target list only
    videos = add_transcripts(
        videos=videos, 
        target_video_ids=new_video_ids, 
        target_languages=target_languages,
        output_file=output_file,
        source_target=source_target,
        cookies_file=args.cookies,
        base_delay=args.delay,
        export_format=export_format,
        workers=workers,
        transcribe_missing=args.transcribe_missing
    )

    # 7. Post-scrape: Generate RAG dataset ONCE (not after every video)
    if rag_mode:
        print("\n📄 Generating RAG dataset...")
        export_rag_jsonl(output_file, videos)
        base_name = os.path.splitext(output_file)[0]
        print(f"    Saved to: {base_name}_rag.jsonl")

    # 8. Provide Final Summary
    with_transcripts = sum(1 for v in videos if v["transcript"])
    rate_limited = sum(1 for v in videos if v.get("transcript_error") == "rate_limited_429")
    no_subs = sum(1 for v in videos if v.get("transcript_error") == "no_subtitles_found")

    print(f"\n{'=' * 60}")
    print(f" Hoàn thành! Lưu tại: {output_file}")
    print(f"   Tổng số video:        {len(videos)}")
    print(f"   Có phụ đề:            {with_transcripts}")
    print(f"   Bị giới hạn API:      {rate_limited}")
    print(f"   Không có phụ đề:      {no_subs}")
    print(f"{'=' * 60}")
    if rate_limited > 0:
        print(f"\n  {rate_limited} videos were rate limited. Run the script again later to retry them.")
    print(f"{'=' * 60}")

    # 9. Post-Processing: AI Knowledge Base Generation
    if args.knowledge_base:
        if not build_knowledge_base(output_file):
            sys.exit(1)


if __name__ == "__main__":
    main()
