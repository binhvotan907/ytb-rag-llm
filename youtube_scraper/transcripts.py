import os
import re
import time
import subprocess
import tempfile
import concurrent.futures
import threading
from .utils import save_progress

print_lock = threading.Lock()
save_lock = threading.Lock()
_WHISPER_MODEL = None


def clean_caption_text(text):
    """Làm sạch text subtitle."""
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_repeated_phrases(text, max_phrase_len=12):
    """
    Khử các cụm bị lặp liên tiếp kiểu:
    'abc abc abc' hoặc các cụm nhiều từ bị lặp nối tiếp.
    """
    words = text.split()
    if not words:
        return text

    changed = True
    while changed:
        changed = False
        n = len(words)

        for size in range(min(max_phrase_len, n // 2), 0, -1):
            new_words = []
            i = 0

            while i < len(words):
                phrase = words[i:i + size]
                j = i + size

                repeat_count = 1
                while j + size <= len(words) and words[j:j + size] == phrase:
                    repeat_count += 1
                    j += size

                new_words.extend(phrase)
                if repeat_count > 1:
                    changed = True

                i = j

            words = new_words

    return " ".join(words)


def merge_overlap_text(prev_text, curr_text, min_overlap_words=3, max_overlap_words=20):
    """
    Nếu curr_text bắt đầu bằng phần cuối của prev_text thì cắt phần overlap đi.
    """
    prev_words = prev_text.split()
    curr_words = curr_text.split()

    if not prev_words or not curr_words:
        return curr_text

    max_check = min(max_overlap_words, len(prev_words), len(curr_words))

    for k in range(max_check, min_overlap_words - 1, -1):
        if prev_words[-k:] == curr_words[:k]:
            return " ".join(curr_words[k:]).strip()

    return curr_text


def parse_vtt_segments(vtt_path):
    """
    Parse file .vtt thành danh sách subtitle segments, có khử lặp tốt hơn.
    """
    raw_segments = []

    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if not lines:
            continue

        time_line = None
        text_lines = []

        for line in lines:
            if (
                line.startswith("WEBVTT")
                or line.startswith("Kind:")
                or line.startswith("Language:")
            ):
                continue

            if "-->" in line:
                time_line = line
                continue

            if line.isdigit():
                continue

            clean_line = clean_caption_text(line)
            if clean_line:
                text_lines.append(clean_line)

        if not time_line or not text_lines:
            continue

        parts = time_line.split("-->")
        start = parts[0].strip()
        end = parts[1].strip() if len(parts) > 1 else ""

        text = " ".join(text_lines).strip()
        text = remove_repeated_phrases(text)

        if text:
            raw_segments.append({
                "start": start,
                "end": end,
                "text": text
            })

    final_segments = []
    prev_text = ""

    for seg in raw_segments:
        curr_text = clean_caption_text(seg["text"])
        curr_text = remove_repeated_phrases(curr_text)

        if prev_text and curr_text == prev_text:
            continue

        if prev_text:
            curr_text = merge_overlap_text(prev_text, curr_text)

        curr_text = clean_caption_text(curr_text)
        curr_text = remove_repeated_phrases(curr_text)

        if not curr_text:
            continue

        final_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": curr_text
        })
        prev_text = curr_text

    return final_segments


def segments_to_text(segments):
    """Gộp segments thành transcript text sạch hơn."""
    texts = []
    for seg in segments:
        text = clean_caption_text(seg.get("text", ""))
        text = remove_repeated_phrases(text)
        if text:
            texts.append(text)

    return " ".join(texts).strip()


def get_caption_language(filename):
    """Extract the subtitle language code from a yt-dlp VTT filename."""
    parts = filename.split(".")
    if len(parts) >= 3:
        return parts[-2]
    return ""


def sort_vtt_files_by_language(vtt_files, preferred_langs):
    """Prefer captions in the same order as --langs."""
    lang_priority = {lang: index for index, lang in enumerate(preferred_langs)}
    return sorted(
        vtt_files,
        key=lambda filename: lang_priority.get(get_caption_language(filename), len(lang_priority))
    )


def seconds_to_vtt_timestamp(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{milliseconds:03d}"


def get_whisper_model():
    """Load faster-whisper once for audio fallback transcription."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install it with: pip install faster-whisper"
            ) from exc

        model_name = os.getenv("WHISPER_MODEL", "small")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        print(f"   Loading Whisper model ({model_name}, {device}, {compute_type})...")
        _WHISPER_MODEL = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _WHISPER_MODEL


def transcribe_audio_ytdlp(video_id, target_languages, cookies_file=None):
    """
    Download audio with yt-dlp and transcribe it with faster-whisper.

    Returns:
        tuple: (text, segments, language, error)
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["yt-dlp"]

        if cookies_file and os.path.exists(cookies_file):
            cmd.extend(["--cookies", cookies_file])

        cmd.extend([
            "-f", "bestaudio/best",
            "--no-playlist",
            "--ignore-errors",
            "-o", os.path.join(tmpdir, "%(id)s.%(ext)s"),
            url
        ])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
        except subprocess.TimeoutExpired:
            return None, None, None, "audio_download_timeout"
        except Exception as e:
            return None, None, None, f"audio_download_error: {e}"

        output = result.stdout + result.stderr
        if "429" in output:
            return None, None, None, "rate_limited_429"

        audio_files = [
            os.path.join(tmpdir, filename)
            for filename in os.listdir(tmpdir)
            if not filename.endswith((".vtt", ".json", ".part"))
        ]

        if not audio_files:
            return None, None, None, "audio_not_found"

        try:
            model = get_whisper_model()
            language = "vi" if any(lang.startswith("vi") for lang in target_languages) else None
            whisper_segments, info = model.transcribe(
                audio_files[0],
                language=language,
                vad_filter=True
            )

            segments = []
            for segment in whisper_segments:
                text = clean_caption_text(segment.text)
                if not text:
                    continue
                segments.append({
                    "start": seconds_to_vtt_timestamp(segment.start),
                    "end": seconds_to_vtt_timestamp(segment.end),
                    "text": text
                })

            transcript = segments_to_text(segments)
            detected_language = getattr(info, "language", None) or language or "unknown"

            if transcript:
                return transcript, segments, f"{detected_language}-whisper", None

            return None, None, detected_language, "empty_transcription"
        except Exception as e:
            return None, None, None, f"transcription_error: {e}"


def get_transcript_ytdlp(video_id, target_languages, cookies_file=None):
    """
    Fetch transcript using yt-dlp.

    Returns:
        tuple: (text, segments, language, error, was_rate_limited)
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    if isinstance(target_languages, list) and len(target_languages) > 0:
        preferred_langs = target_languages
    else:
        preferred_langs = ["vi-orig", "vi", "vi-VN"]

    langs_str = ",".join(preferred_langs)

    with tempfile.TemporaryDirectory() as tmpdir:
        for sub_type in ["--write-auto-subs", "--write-subs"]:
            cmd = [
                "yt-dlp",
            ]

            if cookies_file and os.path.exists(cookies_file):
                cmd.extend(["--cookies", cookies_file])

            cmd.extend([
                sub_type,
                "--sub-langs", langs_str,
                "--sub-format", "vtt",
                "--skip-download",
                "--no-playlist",
                "--ignore-errors",
                "-o", os.path.join(tmpdir, "%(id)s.%(ext)s"),
                url
            ])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                output = result.stdout + result.stderr

                vtt_files = sort_vtt_files_by_language(
                    [f for f in os.listdir(tmpdir) if f.endswith(".vtt")],
                    preferred_langs
                )

                if vtt_files:
                    vtt_path = os.path.join(tmpdir, vtt_files[0])

                    segments = parse_vtt_segments(vtt_path)
                    text = segments_to_text(segments)

                    if text:
                        lang = get_caption_language(vtt_files[0]) or "unknown"
                        return text, segments, lang, None, False

                if "429" in output:
                    return None, None, None, "rate_limited_429", True

            except subprocess.TimeoutExpired:
                return None, None, None, "timeout", False
            except Exception as e:
                return None, None, None, str(e), False

    return None, None, None, "no_subtitles_found", False


def add_transcripts(
    videos,
    target_video_ids,
    target_languages,
    output_file,
    source_target,
    cookies_file=None,
    base_delay=4,
    max_delay=60,
    backoff_multiplier=2,
    export_format="json",
    workers=1,
    save_every=5,
    transcribe_missing=False
):
    """
    Add transcripts only for specified videos with progressive backoff.

    Args:
        save_every (int): Save progress after every N processed videos.
    """
    videos_to_process = [v for v in videos if v["id"] in target_video_ids]
    total = len(videos_to_process)
    already_done = sum(1 for v in videos_to_process if v.get("transcript") is not None)

    if total == 0:
        return videos

    print(f"Fetching transcripts for {total} targeted videos using yt-dlp...")
    print(f"Already have: {already_done} | Still needed: {total - already_done}")
    print(f"Workers: {workers} | Base delay: {base_delay}s between videos. Will back off on 429s.\n")

    rate_limit_state = {"current_delay": base_delay, "consecutive_429s": 0}
    delay_lock = threading.Lock()
    progress_counter = {"processed": 0}

    def maybe_save(force=False):
        with save_lock:
            if force or (progress_counter["processed"] % save_every == 0 and progress_counter["processed"] > 0):
                save_progress(output_file, source_target, videos, export_format=export_format)

    def process_video(args):
        idx, video = args

        if video.get("transcript") is not None:
            with print_lock:
                print(f"  [{idx+1}/{total}] SKIP (have transcript): {video['title'][:50]}")
            return

        video["transcript_error"] = None
        video_id = video["id"]
        title = video["title"][:55]

        with delay_lock:
            local_delay = rate_limit_state["current_delay"]

        if local_delay > 0:
            time.sleep(local_delay)

        text, segments, lang, error, was_429 = get_transcript_ytdlp(
            video_id,
            target_languages=target_languages,
            cookies_file=cookies_file
        )

        if transcribe_missing and error == "no_subtitles_found":
            with print_lock:
                print(f"  [{idx+1}/{total}] No subtitles. Transcribing audio with faster-whisper: {title}")

            text, segments, lang, whisper_error = transcribe_audio_ytdlp(
                video_id,
                target_languages=target_languages,
                cookies_file=cookies_file
            )
            error = whisper_error
            was_429 = whisper_error == "rate_limited_429"

        with delay_lock:
            if was_429:
                rate_limit_state["consecutive_429s"] += 1
                rate_limit_state["current_delay"] = min(
                    rate_limit_state["current_delay"] * backoff_multiplier,
                    max_delay
                )
                video["transcript_error"] = "rate_limited_429"

                with print_lock:
                    print(f"  [{idx+1}/{total}] ⚠ 429 RATE LIMITED: {title}")
                    print(
                        f"      → Backing off. New delay: {rate_limit_state['current_delay']}s "
                        f"(hit {rate_limit_state['consecutive_429s']}x in a row)"
                    )
            else:
                if rate_limit_state["consecutive_429s"] > 0:
                    rate_limit_state["current_delay"] = max(
                        int(rate_limit_state["current_delay"] / backoff_multiplier),
                        base_delay
                    )
                    rate_limit_state["consecutive_429s"] = 0

                video["transcript"] = text
                video["transcript_segments"] = segments if segments else []
                video["transcript_language"] = lang
                video["transcript_error"] = error

                status = f"✓ ({lang})" if text else f"✗ {error}"
                with print_lock:
                    print(f"  [{idx+1}/{total}] {status}: {title}")

        with save_lock:
            progress_counter["processed"] += 1

        maybe_save(force=False)

    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_video, args) for args in enumerate(videos_to_process)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    with print_lock:
                        print(f" Worker Thread Exception: {e}")
    else:
        for args in enumerate(videos_to_process):
            process_video(args)

    maybe_save(force=True)
    return videos
