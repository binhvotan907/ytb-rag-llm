import json
import os
import re
import csv

def clean_youtube_input(url):
    url = url.strip()

    if "youtu.be/" in url:
        return "video", url.split("youtu.be/")[1].split("?")[0].split("&")[0].split("/")[0]

    if "watch?v=" in url:
        match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        if match:
            return "video", match.group(1)

    if "/shorts/" in url:
        return "video", url.split("/shorts/")[1].split("?")[0].split("&")[0].split("/")[0]

    unsupported_patterns = (
        "list=",
        "/playlist",
        "/channel/",
        "/c/",
        "/user/",
        "/@",
    )
    if any(pattern in url for pattern in unsupported_patterns):
        return "unsupported", url

    clean_id = url.split("?")[0].split("&")[0].strip()
    return None, clean_id


def count_words(text):
    """Đếm số từ trong text."""
    if not text:
        return 0
    return len(text.split())


def vtt_time_to_seconds(time_str):
    """
    Chuyển timestamp VTT sang giây.
    Ví dụ:
    00:01:05.300 -> 65.3
    01:02:03.000 -> 3723
    """
    if not time_str:
        return 0

    # VTT đôi khi có thêm phần setting phía sau timestamp
    # ví dụ: 00:00:01.000 align:start position:0%
    time_str = time_str.strip().split(" ")[0]

    parts = time_str.split(":")

    try:
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s

        if len(parts) == 2:
            m = int(parts[0])
            s = float(parts[1])
            return m * 60 + s

    except ValueError:
        return 0

    return 0


def seconds_to_hhmmss(seconds):
    """Chuyển số giây sang HH:MM:SS."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_overlap_segments(segments, overlap_size):
    """
    Lấy các segment cuối của chunk trước để overlap sang chunk sau.
    overlap_size tính theo số từ.
    """
    overlap_segments = []
    overlap_words = 0

    for segment in reversed(segments):
        overlap_segments.insert(0, segment)
        overlap_words += count_words(segment.get("text", ""))

        if overlap_words >= overlap_size:
            break

    return overlap_segments


def build_chunk_from_segments(video, chunk_index, segments):
    """
    Tạo một chunk từ nhiều transcript segment liên tiếp.
    """
    video_id = video["id"]
    text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip())
    text = re.sub(r"\s+", " ", text).strip()

    start_raw = segments[0].get("start", "")
    end_raw = segments[-1].get("end", "")

    start_seconds = vtt_time_to_seconds(start_raw)
    end_seconds = vtt_time_to_seconds(end_raw)

    return {
        "video_id": video_id,
        "title": video.get("title", ""),
        "channel": video.get("channel_title", ""),
        "chunk_id": f"{video_id}_chunk_{chunk_index:04d}",
        "chunk_index": chunk_index,
        "start_time": seconds_to_hhmmss(start_seconds),
        "end_time": seconds_to_hhmmss(end_seconds),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "url": video.get("url", f"https://www.youtube.com/watch?v={video_id}"),
        "url_with_timestamp": f"https://www.youtube.com/watch?v={video_id}&t={int(start_seconds)}s",
        "text": text,
        "word_count": count_words(text)
    }


def chunk_transcript(video, chunk_size=300, overlap_size=60):
    """
    Chia transcript thành chunks dùng cho RAG.

    Cải tiến:
    - Ưu tiên chia theo transcript_segments để giữ timestamp.
    - Không cắt cứng theo ký tự.
    - Gom nhiều subtitle segment thành một chunk.
    - Có overlap giữa các chunk để tránh mất ý.
    - Mỗi chunk có start_time, end_time và url_with_timestamp.
    """

    segments = video.get("transcript_segments", [])

    # Nếu video chưa có transcript_segments thì fallback về transcript text cũ
    if not segments:
        text = video.get("transcript", "")
        if not text:
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        segments = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                segments.append({
                    "start": "",
                    "end": "",
                    "text": sentence
                })

    chunks = []
    current_segments = []
    current_words = 0
    chunk_index = 1

    for segment in segments:
        segment_text = segment.get("text", "").strip()
        if not segment_text:
            continue

        segment_words = count_words(segment_text)

        if current_words + segment_words <= chunk_size:
            current_segments.append(segment)
            current_words += segment_words
        else:
            if current_segments:
                chunk = build_chunk_from_segments(
                    video=video,
                    chunk_index=chunk_index,
                    segments=current_segments
                )
                chunks.append(chunk)
                chunk_index += 1

            overlap_segments = get_overlap_segments(
                current_segments,
                overlap_size=overlap_size
            )

            current_segments = overlap_segments + [segment]
            current_words = count_words(
                " ".join(seg.get("text", "") for seg in current_segments)
            )

    if current_segments:
        chunk = build_chunk_from_segments(
            video=video,
            chunk_index=chunk_index,
            segments=current_segments
        )
        chunks.append(chunk)

    # Gắn liên kết chunk trước/sau để sau này search có thể mở rộng context
    for i, chunk in enumerate(chunks):
        chunk["previous_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
        chunk["next_chunk_id"] = chunks[i + 1]["chunk_id"] if i < len(chunks) - 1 else None

    return chunks


def ensure_parent_dir(file_path):
    """Tạo thư mục cha nếu chưa tồn tại."""
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def export_json(output_file, output_dict):
    """Export standard JSON."""
    ensure_parent_dir(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)


def export_rag_jsonl(output_file, videos):
    """Exports a chunked RAG dataset to _rag.jsonl"""
    base_name = os.path.splitext(output_file)[0]
    rag_file = f"{base_name}_rag.jsonl"

    ensure_parent_dir(rag_file)

    with open(rag_file, "w", encoding="utf-8") as f:
        for video in videos:
            if video.get("transcript"):
                chunks = chunk_transcript(video)
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def export_csv(output_file, output_dict):
    """Export videos to CSV."""
    videos = output_dict.get("videos", [])
    if not videos:
        return
        
    keys_to_write = ["id", "title", "description", "published_at", "channel_title", "tags",
                     "thumbnail_url", "duration", "view_count", "like_count", "comment_count", 
                     "url", "transcript_language", "transcript_error", "transcript"]
                     
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys_to_write, extrasaction='ignore')
        writer.writeheader()
        for video in videos:
            row = video.copy()
            if "tags" in row and isinstance(row["tags"], list):
                row["tags"] = ", ".join(row["tags"])
            if row.get("transcript"):
                # Standardize newline spaces for cleaner CSVs
                row["transcript"] = row["transcript"].replace("\n", " ").replace("\r", "")
            writer.writerow(row)


def export_parquet(output_file, output_dict):
    """Export to Parquet using Pandas."""
    try:
        import pandas as pd
        videos = output_dict.get("videos", [])
        if not videos:
            return
            
        df = pd.DataFrame(videos)
        
        # Convert any list types to string to avoid Parquet type nesting issues sometimes
        if "tags" in df.columns:
            df["tags"] = df["tags"].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)
            
        df.to_parquet(output_file, engine="pyarrow", index=False)
    except ImportError:
        print("\n⚠️ pandas and pyarrow are required for parquet export. Install them with: pip install pandas pyarrow")


def load_existing_progress(output_file, export_format="json"):
    """Load existing JSON file to resume from where we left off.
    
    We ALWAYS read from the internal `.json` tracker regardless of the chosen --format.
    """
    json_path = output_file if export_format == "json" else f"{os.path.splitext(output_file)[0]}.json"
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            videos = data.get("videos", [])
            done = sum(1 for v in videos if v["transcript"] is not None)
            pending = len(videos) - done
            if len(videos) > 0:
                print(f"\n Found existing progress: {done}/{len(videos)} transcripts successfully fetched.")
                print(f"   {pending} videos still need transcripts. Resuming...\n")
            return videos
        except json.JSONDecodeError:
            print(f"\n WARNING: {json_path} is empty or corrupted. Starting fresh.")
            return []
    return []


def save_progress(output_file, source_target, videos, export_format="json"):
    """Save current progress to JSON, plus format-specific outputs.
    
    Note: RAG generation is NOT triggered here. It runs once after scraping
    finishes, inside main.py, to avoid rebuilding the file after every video.
    
    Args:
        output_file (str): The path to the requested output file.
        source_target (str): The ID of the targeted entity.
        videos (list): The video dictionaries.
        export_format (str): json, jsonl, csv, or parquet.
    """
    output = {
        "source_target": source_target,
        "total_videos": len(videos),
        "videos": videos
    }
    
    # 1. ALWAYS save a standard JSON copy as our state tracker
    json_path = output_file if export_format == "json" else f"{os.path.splitext(output_file)[0]}.json"
    export_json(json_path, output)
    
    # 2. Export to requested format
    if export_format == "jsonl":
        export_jsonl(output_file, output)
    elif export_format == "csv":
        export_csv(output_file, output)
    elif export_format == "parquet":
        export_parquet(output_file, output)

