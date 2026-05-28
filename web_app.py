import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse

from youtube_scraper.chatbot import answer_with_llm
from youtube_scraper.search import is_overview_question, print_video_overview, semantic_search


ROOT_DIR = Path(__file__).resolve().parent
LOCAL_DIR = ROOT_DIR / "local"
LOCAL_DIR.mkdir(exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()


def safe_slug(value):
    value = value.strip()
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{6,})", value)
    if match:
        value = match.group(1)
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return value[:48] or f"video-{int(time.time())}"


def fetch_youtube_title(video_url):
    """Fetch a public YouTube title for naming local dataset folders."""
    try:
        encoded_url = quote(video_url, safe="")
        oembed_url = f"https://www.youtube.com/oembed?url={encoded_url}&format=json"
        req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return (payload.get("title") or "").strip()
    except Exception:
        return ""


def json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, socket.timeout):
        return


def read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw or "{}")


def output_paths(output_file):
    output = Path(output_file)
    base = output.with_suffix("")
    return {
        "json": str(output),
        "rag": str(base) + "_rag.jsonl",
        "index": str(base) + "_vector_index.faiss",
        "embeddings": str(base) + "_embeddings.npy",
    }


def create_job(video_url, output_name, delay, transcribe_missing=False):
    job_id = uuid.uuid4().hex[:12]
    title = output_name.strip() if output_name else fetch_youtube_title(video_url)
    dataset_slug = safe_slug(title or video_url)
    dataset_dir = LOCAL_DIR / dataset_slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(dataset_dir / f"{dataset_slug}.json")
    command = [
        sys.executable,
        "-m",
        "youtube_scraper.main",
        "--video",
        video_url,
        "--output",
        output_file,
        "--langs",
        "vi-orig,vi,vi-VN",
        "--delay",
        str(delay),
        "--rag",
        "--knowledge-base",
    ]

    if transcribe_missing:
        command.append("--transcribe-missing")

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "command": " ".join(command),
            "output_file": output_file,
            "log": [],
            "started_at": time.time(),
            "finished_at": None,
            "returncode": None,
        }

    thread = threading.Thread(target=run_job, args=(job_id, command), daemon=True)
    thread.start()
    return JOBS[job_id]


def append_job_log(job_id, line):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job["log"].append(line.rstrip())
            job["log"] = job["log"][-500:]


def run_job(job_id, command):
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            append_job_log(job_id, line)

        returncode = process.wait()
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["returncode"] = returncode
            job["finished_at"] = time.time()
            job["status"] = "completed" if returncode == 0 else "failed"
    except Exception as exc:
        append_job_log(job_id, f"ERROR: {exc}")
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "failed"
            job["finished_at"] = time.time()
            job["returncode"] = -1


def list_datasets():
    datasets = []
    json_files = list(LOCAL_DIR.glob("*.json")) + list(LOCAL_DIR.glob("*/*.json"))
    json_files = sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True)

    for json_file in json_files:
        paths = output_paths(str(json_file))
        title = json_file.stem
        display_name = str(json_file.relative_to(LOCAL_DIR))

        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            videos = data.get("videos", [])
            if videos:
                video = videos[0]
                title = video.get("title") or title
                language = video.get("transcript_language") or "-"
                channel = video.get("channel_title") or ""
                url = video.get("url") or ""
                thumbnail = video.get("thumbnail_url") or ""
                duration = video.get("duration") or "-"
                words = len((video.get("transcript") or "").split())
            else:
                language = "-"
                channel = ""
                url = ""
                thumbnail = ""
                duration = "-"
                words = 0
        except Exception:
            language = "-"
            channel = ""
            url = ""
            thumbnail = ""
            duration = "-"
            words = 0
            pass

        chunk_count = 0
        try:
            with open(paths["rag"], "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunk_count += 1
        except Exception:
            chunk_count = 0

        datasets.append(
            {
                "file": str(json_file),
                "name": display_name,
                "title": title,
                "channel": channel,
                "url": url,
                "thumbnail": thumbnail,
                "duration": duration,
                "language": language,
                "chunks": chunk_count,
                "words": words,
                "has_rag": Path(paths["rag"]).exists(),
                "has_index": Path(paths["index"]).exists(),
            }
        )
    return datasets


def answer_question(output_file, question, use_llm):
    paths = output_paths(output_file)
    if is_overview_question(question) and not use_llm:
        return {
            "mode": "overview",
            "answer": build_metadata_overview(output_file),
            "sources": [],
        }

    if use_llm:
        answer = answer_with_llm(
            question=question,
            output_file=output_file,
            index_file=paths["index"],
            rag_file=paths["rag"],
            top_k=3,
        )
        return {"mode": "llm", "answer": answer, "sources": []}

    results = semantic_search(question, paths["index"], paths["rag"], top_k=3)
    return {
        "mode": "search",
        "answer": "",
        "sources": [
            {
                "rank": item.get("rank"),
                "title": item.get("title"),
                "time": f"{item.get('start_time', '')} - {item.get('end_time', '')}",
                "url": item.get("url_with_timestamp"),
                "text": item.get("text", "")[:900],
                "score": item.get("score"),
                "rerank_score": item.get("rerank_score"),
            }
            for item in results
        ],
    }


def build_metadata_overview(output_file):
    try:
        data = json.loads(Path(output_file).read_text(encoding="utf-8"))
        video = (data.get("videos") or [{}])[0]
    except Exception:
        return "Không đọc được file dữ liệu video."

    parts = [
        f"Tiêu đề: {video.get('title', '')}",
        f"Kênh: {video.get('channel_title', '')}",
        f"Thời lượng: {video.get('duration', '')}",
        f"Link: {video.get('url', '')}",
    ]

    description = (video.get("description") or "").split("---------------------------------")[0].strip()
    if description:
        parts.append("\nMô tả:\n" + description[:1200])

    transcript = (video.get("transcript") or "").strip()
    if transcript:
        parts.append("\nMở đầu transcript:\n" + transcript[:900] + "...")
    return "\n".join(parts)


INDEX_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YouTube RAG</title>
  <style>
    :root {
      color-scheme: dark;
      --ink: #f6f8fb;
      --muted: #9ca8ba;
      --line: rgba(165, 180, 203, 0.2);
      --panel: rgba(14, 18, 27, 0.82);
      --panel-strong: rgba(18, 24, 36, 0.96);
      --field: rgba(7, 10, 16, 0.78);
      --brand: #34d399;
      --brand-2: #38bdf8;
      --accent: #f59e0b;
      --warn: #fbbf24;
      --ok: #34d399;
      --danger: #fb7185;
      --white: #ffffff;
    }
    * { box-sizing: border-box; }
    html { min-height: 100%; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 18% -10%, rgba(52, 211, 153, 0.2), transparent 32%),
        radial-gradient(circle at 86% 0%, rgba(245, 158, 11, 0.12), transparent 28%),
        radial-gradient(circle at 72% 92%, rgba(56, 189, 248, 0.12), transparent 34%),
        linear-gradient(135deg, #07090f 0%, #10131b 46%, #090b10 100%);
      min-height: 100vh;
      overflow-x: hidden;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(148, 163, 184, 0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.06) 1px, transparent 1px);
      background-size: 44px 44px;
      mask-image: linear-gradient(to bottom, rgba(0,0,0,0.88), transparent 82%);
    }
    #neuralCanvas {
      position: fixed;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      opacity: 0.42;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 3;
      height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 30px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(7, 9, 15, 0.86);
      backdrop-filter: blur(18px);
      box-shadow: 0 14px 42px rgba(0, 0, 0, 0.28);
    }
    header h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 800;
      letter-spacing: 0;
      display: flex;
      gap: 10px;
      align-items: center;
    }
    header h1::before {
      content: "";
      width: 16px;
      height: 16px;
      border-radius: 7px;
      background: var(--brand);
      box-shadow: 0 0 0 7px rgba(52, 211, 153, 0.1), 0 0 28px rgba(52, 211, 153, 0.74);
      animation: pulse 2.4s ease-in-out infinite;
    }
    header span {
      color: var(--muted);
      font-size: 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(21, 27, 40, 0.72);
    }
    main {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 20px;
      min-height: calc(100vh - 72px);
      padding: 20px;
    }
    aside {
      border: 1px solid var(--line);
      background: rgba(9, 12, 20, 0.72);
      backdrop-filter: blur(18px);
      border-radius: 18px;
      padding: 18px;
      overflow: auto;
      box-shadow: 0 22px 70px rgba(0, 0, 0, 0.34);
    }
    section {
      overflow: auto;
    }
    .panel {
      position: relative;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 20px 50px rgba(0,0,0,0.24);
      overflow: hidden;
    }
    .panel::after {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      border-radius: inherit;
      background: linear-gradient(120deg, transparent, rgba(52, 211, 153, 0.07), transparent);
      transform: translateX(-120%);
      animation: sweep 8s ease-in-out infinite;
    }
    .panel h2 {
      position: relative;
      font-size: 12px;
      text-transform: uppercase;
      color: #d6deea;
      letter-spacing: 0.1em;
      margin: 0 0 14px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid rgba(148, 163, 184, 0.28);
      border-radius: 12px;
      padding: 13px 14px;
      font: inherit;
      background: var(--field);
      color: var(--ink);
      outline: none;
      transition: border-color 150ms ease, box-shadow 150ms ease, transform 150ms ease;
    }
    input:focus, textarea:focus {
      border-color: rgba(52, 211, 153, 0.76);
      box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.1);
    }
    input::placeholder, textarea::placeholder { color: #66798c; }
    textarea { min-height: 126px; resize: vertical; }
    button {
      border: 0;
      border-radius: 12px;
      padding: 13px 14px;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
      color: #fff;
      background: linear-gradient(135deg, #059669, #0284c7);
      box-shadow: 0 14px 30px rgba(5, 150, 105, 0.22);
      transition: transform 150ms ease, filter 150ms ease, box-shadow 150ms ease;
    }
    button:hover {
      transform: translateY(-1px);
      filter: brightness(1.08);
      box-shadow: 0 18px 38px rgba(52, 211, 153, 0.22);
    }
    button:active { transform: translateY(0); }
    button.secondary {
      min-height: 54px;
      background: linear-gradient(135deg, #059669, #2563eb 58%, #7c3aed);
      box-shadow: 0 18px 44px rgba(37, 99, 235, 0.26);
    }
    button.ghost {
      background: rgba(148, 163, 184, 0.12);
      color: #dceafe;
      border: 1px solid rgba(148, 163, 184, 0.22);
      box-shadow: none;
    }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { flex: 1; }
    .stack { display: grid; gap: 10px; }
    details.panel { padding: 0; }
    details.panel summary {
      position: relative;
      list-style: none;
      cursor: pointer;
      padding: 18px;
      font-size: 12px;
      text-transform: uppercase;
      color: #d6deea;
      letter-spacing: 0.1em;
      font-weight: 800;
    }
    details.panel summary::-webkit-details-marker { display: none; }
    details.panel summary::after {
      content: "+";
      position: absolute;
      right: 18px;
      top: 15px;
      color: var(--muted);
      font-size: 20px;
      line-height: 1;
    }
    details.panel[open] summary::after { content: "-"; }
    details.panel .details-body { padding: 0 18px 18px; }
    .dataset {
      position: relative;
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 12px;
      cursor: pointer;
      background: rgba(13, 18, 28, 0.72);
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
      overflow: hidden;
    }
    .dataset:hover {
      transform: translateY(-1px);
      border-color: rgba(52, 211, 153, 0.54);
      background: rgba(18, 29, 42, 0.86);
    }
    .dataset.active {
      border-color: var(--brand);
      background: rgba(17, 45, 42, 0.68);
      box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.1);
    }
    .dataset strong { display: block; font-size: 13px; line-height: 1.4; }
    .dataset small {
      display: block;
      color: var(--muted);
      line-height: 1.45;
      margin-top: 6px;
      overflow-wrap: anywhere;
    }
    .badge {
      display: inline-block;
      margin-top: 8px;
      padding: 3px 7px;
      border-radius: 999px;
      font-size: 11px;
      color: var(--ok);
      background: rgba(52, 211, 153, 0.14);
      border: 1px solid rgba(52, 211, 153, 0.22);
    }
    .badge.warn {
      color: var(--warn);
      background: rgba(251, 191, 36, 0.13);
      border-color: rgba(251, 191, 36, 0.22);
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.45 Consolas, "Courier New", monospace;
      color: #cae7ff;
      background: rgba(2, 6, 12, 0.82);
      border: 1px solid rgba(148, 163, 184, 0.16);
      border-radius: 14px;
      padding: 12px;
      min-height: 140px;
      max-height: 300px;
      overflow: auto;
    }
    .answer, .source {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      margin: 0 0 18px;
      white-space: pre-wrap;
      line-height: 1.62;
      box-shadow: 0 18px 42px rgba(0,0,0,0.22);
    }
    .answer {
      border-color: rgba(56, 189, 248, 0.28);
      background: linear-gradient(180deg, rgba(17, 27, 40, 0.96), rgba(10, 14, 22, 0.92));
    }
    .answer ul {
      margin: 8px 0 14px;
      padding-left: 22px;
    }
    .answer li {
      margin: 5px 0;
    }
    .answer p {
      margin: 0 0 12px;
    }
    .source h3 { margin: 0 0 8px; font-size: 14px; }
    .source a { color: #93c5fd; text-decoration: none; }
    .empty {
      color: var(--muted);
      text-align: center;
      padding: 56px 16px;
    }
    .details-body .empty { padding: 18px 12px; }
    .hero-note {
      position: relative;
      color: var(--muted);
      line-height: 1.5;
      font-size: 13px;
      margin: -2px 0 14px;
    }
    .status-strip {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 14px;
    }
    .status-pill {
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 12px;
      padding: 10px;
      background: rgba(7, 10, 16, 0.48);
      color: var(--muted);
      font-size: 11px;
      text-align: center;
    }
    .status-pill strong {
      display: block;
      color: var(--ink);
      font-size: 13px;
      margin-bottom: 3px;
    }
    @keyframes pulse {
      0%, 100% { opacity: 0.72; transform: scale(1); }
      50% { opacity: 1; transform: scale(1.12); }
    }
    @keyframes sweep {
      0%, 42%, 100% { transform: translateX(-120%); }
      56% { transform: translateX(120%); }
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; padding: 12px; }
      aside { border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }
      header { padding: 0 16px; }
    }
  </style>
</head>
<body>
  <canvas id="neuralCanvas" aria-hidden="true"></canvas>
  <header>
    <h1>YouTube RAG Studio</h1>
    <span>Semantic video QA</span>
  </header>
  <main>
    <aside>
      <div class="panel">
        <h2>Nguồn video</h2>
        <p class="hero-note">Dán liên kết YouTube để tạo bộ tri thức cho video.</p>
        <div class="stack">
          <div>
            <label for="videoUrl">YouTube URL</label>
            <input id="videoUrl" placeholder="https://www.youtube.com/watch?v=..." />
          </div>
          <button id="crawlBtn">Xử lý video</button>
          <div class="status-strip" aria-hidden="true">
            <div class="status-pill"><strong>01</strong>Crawl</div>
            <div class="status-pill"><strong>02</strong>Vector</div>
            <div class="status-pill"><strong>03</strong>Ask</div>
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>Thư viện</h2>
        <div id="datasets"></div>
      </div>
    </aside>

    <section>
      <div class="panel">
        <h2>Hỏi đáp nội dung</h2>
        <div class="stack">
          <textarea id="question" placeholder="Nhập câu hỏi về video..."></textarea>
          <button id="askBtn" class="secondary">Trả lời thông minh</button>
        </div>
      </div>

      <div id="result" class="empty">Chọn một video trong thư viện hoặc xử lý video mới.</div>

      <details class="panel" id="retrievalPanel">
        <summary>Kiểm chứng truy xuất</summary>
        <div class="details-body stack">
          <button id="searchBtn" class="ghost">Cập nhật đoạn liên quan</button>
          <div id="sourcesResult" class="empty">Chưa có kết quả truy xuất.</div>
        </div>
      </details>

      <details class="panel" id="logPanel">
        <summary>Nhật ký hệ thống</summary>
        <div class="details-body">
          <pre id="log">Chưa có job đang chạy.</pre>
        </div>
      </details>
    </section>
  </main>

  <script>
    let selectedFile = "";
    let currentJob = "";
    let pollTimer = null;
    let lastSmartQuestion = "";
    let lastSmartDataset = "";
    let lastRetrievalQuestion = "";
    let lastRetrievalDataset = "";

    const $ = (id) => document.getElementById(id);

    function startNeuralCanvas() {
      const canvas = $("neuralCanvas");
      const ctx = canvas.getContext("2d");
      let width = 0;
      let height = 0;
      let nodes = [];
      let frame = 0;

      function resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = width + "px";
        canvas.style.height = height + "px";
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const count = Math.max(34, Math.min(86, Math.floor((width * height) / 18500)));
        nodes = Array.from({ length: count }, (_, index) => ({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.28,
          vy: (Math.random() - 0.5) * 0.28,
          r: 1.3 + Math.random() * 1.8,
          phase: Math.random() * Math.PI * 2,
          hue: index % 3
        }));
      }

      function draw() {
        frame += 0.012;
        ctx.clearRect(0, 0, width, height);

        for (const node of nodes) {
          node.x += node.vx;
          node.y += node.vy;
          if (node.x < -20) node.x = width + 20;
          if (node.x > width + 20) node.x = -20;
          if (node.y < -20) node.y = height + 20;
          if (node.y > height + 20) node.y = -20;
        }

        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const a = nodes[i];
            const b = nodes[j];
            const dx = a.x - b.x;
            const dy = a.y - b.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 150) {
              const alpha = (1 - dist / 150) * 0.23;
              ctx.strokeStyle = `rgba(96, 165, 250, ${alpha})`;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(a.x, a.y);
              ctx.lineTo(b.x, b.y);
              ctx.stroke();
            }
          }
        }

        for (const node of nodes) {
          const glow = 0.52 + Math.sin(frame * 2 + node.phase) * 0.22;
          ctx.beginPath();
          ctx.fillStyle = node.hue === 0
            ? `rgba(45, 212, 191, ${glow})`
            : node.hue === 1
              ? `rgba(96, 165, 250, ${glow})`
              : `rgba(245, 158, 11, ${glow * 0.75})`;
          ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
          ctx.fill();
        }

        requestAnimationFrame(draw);
      }

      window.addEventListener("resize", resize);
      resize();
      draw();
    }

    async function api(path, options = {}) {
      const res = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    function renderDatasets(items) {
      const root = $("datasets");
      root.innerHTML = "";
      if (!items.length) {
        root.innerHTML = "<div class='empty'>Chưa có video nào.</div>";
        return;
      }
      items.forEach(item => {
        const div = document.createElement("div");
        div.className = "dataset" + (item.file === selectedFile ? " active" : "");
        div.innerHTML = `
          <strong>${escapeHtml(item.title)}</strong>
          <small>${escapeHtml(item.name)}</small><br>
          <span class="badge ${item.has_index ? "" : "warn"}">${item.has_index ? "Sẵn sàng hỏi đáp" : "Đang thiếu index"}</span>
        `;
        div.onclick = () => {
          selectedFile = item.file;
          lastSmartQuestion = "";
          lastSmartDataset = "";
          lastRetrievalQuestion = "";
          lastRetrievalDataset = "";
          renderDatasets(items);
          $("result").className = "empty";
          $("result").textContent = "Đã chọn: " + item.title;
          $("sourcesResult").className = "empty";
          $("sourcesResult").textContent = "Chưa có kết quả truy xuất.";
        };
        root.appendChild(div);
      });
    }

    async function refreshDatasets() {
      const data = await api("/api/datasets");
      renderDatasets(data.datasets);
    }

    function pollJob() {
      if (!currentJob) return;
      api("/api/jobs/" + currentJob).then(job => {
        $("log").textContent = job.log.length ? job.log.join("\n") : "Đang khởi động pipeline...";
        $("log").scrollTop = $("log").scrollHeight;
        if (job.status !== "running") {
          clearInterval(pollTimer);
          selectedFile = job.output_file;
          refreshDatasets();
        }
      }).catch(err => {
        $("log").textContent = err.message;
      });
    }

    $("crawlBtn").onclick = async () => {
      const videoUrl = $("videoUrl").value.trim();
      if (!videoUrl) return alert("Nhập YouTube URL trước.");
      $("log").textContent = "Đang tạo phiên xử lý...";
      const job = await api("/api/jobs", {
        method: "POST",
        body: JSON.stringify({
          video_url: videoUrl,
          output_name: "",
          delay: 15
        })
      });
      currentJob = job.id;
      clearInterval(pollTimer);
      pollTimer = setInterval(pollJob, 1500);
      pollJob();
    };

    async function ask(useLlm) {
      if (!selectedFile) return alert("Chọn dataset trước.");
      const question = $("question").value.trim();
      if (!question) return alert("Nhập câu hỏi trước.");

      if (
        useLlm
        && question === lastSmartQuestion
        && selectedFile === lastSmartDataset
        && !$("result").classList.contains("empty")
      ) {
        return;
      }

      if (
        !useLlm
        && question === lastRetrievalQuestion
        && selectedFile === lastRetrievalDataset
        && !$("sourcesResult").classList.contains("empty")
      ) {
        $("retrievalPanel").open = true;
        return;
      }

      const target = useLlm ? $("result") : $("sourcesResult");
      target.className = "answer";
      target.textContent = useLlm ? "Đang trả lời..." : "Đang tìm đoạn liên quan...";
      const data = await api("/api/ask", {
        method: "POST",
        body: JSON.stringify({ output_file: selectedFile, question, use_llm: useLlm })
      });
      renderResult(data);

      if (useLlm) {
        lastSmartQuestion = question;
        lastSmartDataset = selectedFile;
      } else {
        lastRetrievalQuestion = question;
        lastRetrievalDataset = selectedFile;
        if (!$("retrievalPanel").open) {
          $("retrievalPanel").open = true;
        }
      }
    }

    $("askBtn").onclick = () => ask(true);
    $("searchBtn").onclick = () => ask(false);

    $("retrievalPanel").addEventListener("toggle", () => {
      if (!$("retrievalPanel").open) return;
      const question = $("question").value.trim();
      if (!selectedFile || !question) return;
      if (question === lastRetrievalQuestion && selectedFile === lastRetrievalDataset) return;
      ask(false);
    });

    function renderResult(data) {
      const root = data.mode === "search" ? $("sourcesResult") : $("result");
      root.className = "";
      if (data.answer) {
        root.innerHTML = `<div class="answer">${formatAnswer(data.answer)}</div>`;
      } else if (data.sources && data.sources.length) {
        root.innerHTML = data.sources.map(src => `
          <div class="source">
            <h3>#${src.rank} ${escapeHtml(src.title || "")}</h3>
            <div><strong>${escapeHtml(src.time || "")}</strong></div>
            <div><a href="${src.url}" target="_blank">${escapeHtml(src.url || "")}</a></div>
            <p>${escapeHtml(src.text || "")}</p>
          </div>
        `).join("");
      } else {
        root.className = "empty";
        root.textContent = data.mode === "search"
          ? "Chưa tìm thấy đoạn liên quan."
          : "Không có kết quả.";
      }
    }

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function formatInlineMarkdown(value) {
      return escapeHtml(value)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<strong>$1</strong>");
    }

    function formatAnswer(value) {
      const lines = String(value || "").split(/\r?\n/);
      const parts = [];
      let listItems = [];

      function flushList() {
        if (!listItems.length) return;
        parts.push("<ul>" + listItems.map(item => `<li>${formatInlineMarkdown(item)}</li>`).join("") + "</ul>");
        listItems = [];
      }

      for (const line of lines) {
        const trimmed = line.trim();
        const bullet = trimmed.match(/^[-*]\s+(.+)$/);
        if (bullet) {
          listItems.push(bullet[1]);
          continue;
        }

        flushList();
        if (trimmed) {
          parts.push(`<p>${formatInlineMarkdown(trimmed)}</p>`);
        }
      }

      flushList();
      return parts.join("");
    }

    startNeuralCanvas();
    refreshDatasets();
  </script>
</body>
</html>
"""


NEW_INDEX_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YouTube RAG Studio</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">YR</div>
        <div>
          <h1>YouTube RAG Studio</h1>
          <p>Video knowledge workspace</p>
        </div>
      </div>

      <form id="ingestForm" class="ingest-panel">
        <label for="videoUrl">YouTube video</label>
        <div class="url-row">
          <input id="videoUrl" name="videoUrl" type="url" placeholder="https://www.youtube.com/watch?v=..." required>
          <button type="submit" title="Xử lý video">Go</button>
        </div>
        <label class="toggle-row">
          <input id="transcribeMissing" type="checkbox">
          <span>Tự transcribe nếu không có phụ đề</span>
        </label>
      </form>

      <div class="job-panel" id="jobPanel" hidden>
        <div class="job-head">
          <span id="jobStage">Queued</span>
          <strong id="jobProgressText">0%</strong>
        </div>
        <div class="progress-track"><div id="jobProgress" class="progress-bar"></div></div>
        <pre id="jobLog"></pre>
      </div>

      <div class="library-head">
        <span>Library</span>
        <button id="refreshVideos" type="button" title="Làm mới">Refresh</button>
      </div>
      <div id="videoList" class="video-list"></div>
    </aside>

    <main class="workspace">
      <section class="video-hero" id="videoHero">
        <div class="empty-state">
          <h2>Chọn một video hoặc gửi link mới</h2>
          <p>App sẽ crawl transcript, build vector index, lưu thành dataset, rồi mở một khung hỏi đáp có nguồn timestamp.</p>
        </div>
      </section>

      <section class="chat-surface">
        <div id="messages" class="messages">
          <div class="message assistant">
            <div class="bubble">Mình sẵn sàng trả lời khi bạn chọn một video đã xử lý xong.</div>
          </div>
        </div>
        <form id="askForm" class="ask-form">
          <input id="questionInput" type="text" placeholder="Hỏi về nội dung video..." disabled>
          <button id="askButton" type="submit" disabled>Ask</button>
        </form>
      </section>

      <section class="support-grid">
        <details class="evidence-panel" id="evidencePanel">
          <summary>Kiểm chứng truy xuất</summary>
          <div class="evidence-actions">
            <button id="evidenceButton" type="button">Cập nhật đoạn liên quan</button>
          </div>
          <div id="evidenceList" class="evidence-list">Chưa có kết quả truy xuất.</div>
        </details>
        <details class="evidence-panel" id="systemLogPanel">
          <summary>Nhật ký hệ thống</summary>
          <pre id="jobLogMirror">Chưa có job đang chạy.</pre>
        </details>
      </section>
    </main>
  </div>
<script src="/static/app.js"></script>
</body>
</html>
"""


APP_CSS = r"""
:root {
  color-scheme: dark;
  --bg: #07090f;
  --panel: rgba(16, 20, 30, .86);
  --panel-2: rgba(20, 27, 39, .92);
  --line: rgba(164, 179, 203, .22);
  --ink: #f5f7fb;
  --muted: #98a4b6;
  --brand: #5ee0a3;
  --blue: #5aa7ff;
  --violet: #7c3aed;
  --warn: #f59e0b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at 18% -8%, rgba(94, 224, 163, .18), transparent 30%),
    radial-gradient(circle at 84% 10%, rgba(245, 158, 11, .11), transparent 26%),
    linear-gradient(135deg, #080b12, #111520 50%, #080a10);
}
body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(148, 163, 184, .055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148, 163, 184, .055) 1px, transparent 1px);
  background-size: 44px 44px;
  mask-image: linear-gradient(to bottom, #000, transparent 88%);
}
.app-shell {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: 380px minmax(0, 1fr);
  min-height: 100vh;
}
.sidebar {
  border-right: 1px solid var(--line);
  padding: 24px;
  background: rgba(8, 11, 18, .74);
  backdrop-filter: blur(18px);
  overflow: auto;
}
.brand {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 24px;
}
.brand-mark {
  width: 46px;
  height: 46px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, var(--brand), var(--blue));
  color: #06100d;
  font-weight: 900;
  box-shadow: 0 16px 38px rgba(94, 224, 163, .2);
}
.brand h1 {
  margin: 0;
  font-size: 22px;
  letter-spacing: 0;
}
.brand p {
  margin: 3px 0 0;
  color: var(--muted);
  font-size: 13px;
}
.ingest-panel,
.job-panel,
.video-list .video-card,
.chat-surface,
.video-hero,
.evidence-panel {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 18px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 18px 54px rgba(0,0,0,.24);
}
.ingest-panel {
  padding: 18px;
  display: grid;
  gap: 12px;
  margin-bottom: 20px;
}
label, .library-head span {
  color: #d5dce8;
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .1em;
}
.url-row {
  display: grid;
  grid-template-columns: 1fr 54px;
  gap: 10px;
}
.toggle-row {
  display: flex;
  gap: 9px;
  align-items: center;
  color: var(--muted);
  text-transform: none;
  letter-spacing: 0;
  font-weight: 650;
  font-size: 13px;
}
.toggle-row input {
  width: auto;
  accent-color: var(--brand);
}
input {
  min-width: 0;
  border: 1px solid rgba(148, 163, 184, .26);
  border-radius: 14px;
  padding: 13px 14px;
  background: rgba(5, 8, 13, .78);
  color: var(--ink);
  font: inherit;
  outline: none;
}
input:focus {
  border-color: rgba(94, 224, 163, .7);
  box-shadow: 0 0 0 4px rgba(94, 224, 163, .09);
}
button {
  border: 0;
  border-radius: 14px;
  padding: 12px 14px;
  background: linear-gradient(135deg, #10b981, #3b82f6 62%, #7c3aed);
  color: white;
  font: inherit;
  font-weight: 800;
  cursor: pointer;
  transition: transform .15s ease, filter .15s ease;
}
button:hover { transform: translateY(-1px); filter: brightness(1.06); }
button:disabled { opacity: .48; cursor: not-allowed; transform: none; }
.job-panel {
  padding: 16px;
  margin-bottom: 20px;
}
.job-head,
.library-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.library-head {
  margin: 8px 0 12px;
}
.library-head button,
.evidence-actions button {
  background: rgba(148, 163, 184, .13);
  border: 1px solid rgba(148, 163, 184, .22);
  box-shadow: none;
}
.progress-track {
  height: 8px;
  border-radius: 999px;
  background: rgba(148, 163, 184, .16);
  margin: 12px 0;
  overflow: hidden;
}
.progress-bar {
  width: 0%;
  height: 100%;
  background: linear-gradient(90deg, var(--brand), var(--blue));
  transition: width .3s ease;
}
pre {
  margin: 0;
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
  color: #cfe2ff;
  font: 12px/1.45 Consolas, "Courier New", monospace;
}
.video-list {
  display: grid;
  gap: 12px;
}
.video-card {
  padding: 14px;
  cursor: pointer;
  transition: transform .15s ease, border-color .15s ease, background .15s ease;
}
.video-card:hover {
  transform: translateY(-1px);
  border-color: rgba(94, 224, 163, .46);
}
.video-card.active {
  border-color: var(--brand);
  background: rgba(18, 48, 43, .68);
  box-shadow: 0 0 0 4px rgba(94, 224, 163, .08);
}
.video-card strong {
  display: block;
  line-height: 1.35;
}
.video-card small {
  display: block;
  color: var(--muted);
  margin-top: 6px;
  overflow-wrap: anywhere;
}
.ready-pill {
  display: inline-block;
  margin-top: 12px;
  padding: 4px 9px;
  border-radius: 999px;
  color: var(--brand);
  background: rgba(94, 224, 163, .13);
  border: 1px solid rgba(94, 224, 163, .22);
  font-size: 12px;
}
.workspace {
  padding: 28px;
  overflow: auto;
}
.video-hero {
  min-height: 228px;
  padding: 28px;
  display: flex;
  align-items: end;
  background:
    linear-gradient(90deg, rgba(8, 11, 18, .9), rgba(8, 11, 18, .48)),
    var(--panel);
  overflow: hidden;
}
.video-hero.has-thumb {
  background-size: cover;
  background-position: center;
}
.video-title {
  max-width: 780px;
}
.video-title h2,
.empty-state h2 {
  margin: 0;
  font-size: clamp(28px, 4vw, 48px);
  line-height: 1.05;
}
.video-title p,
.empty-state p {
  color: var(--muted);
  max-width: 720px;
  line-height: 1.55;
  margin: 14px 0 0;
}
.chat-surface {
  margin-top: 16px;
  padding: 18px;
}
.support-grid {
  display: grid;
  gap: 12px;
  margin-top: 16px;
}
.messages {
  min-height: 290px;
  max-height: 56vh;
  overflow: auto;
  display: grid;
  align-content: start;
  gap: 14px;
  padding: 4px 4px 14px;
}
.message {
  display: flex;
}
.message.user { justify-content: flex-end; }
.bubble {
  max-width: min(760px, 88%);
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, .18);
  line-height: 1.6;
  background: rgba(148, 163, 184, .1);
}
.message.user .bubble {
  background: linear-gradient(135deg, rgba(16, 185, 129, .2), rgba(59, 130, 246, .2));
  border-color: rgba(94, 224, 163, .24);
}
.bubble ul { margin: 8px 0 12px; padding-left: 22px; }
.bubble p { margin: 0 0 10px; }
.ask-form {
  display: grid;
  grid-template-columns: 1fr 100px;
  gap: 10px;
  border-top: 1px solid var(--line);
  padding-top: 14px;
}
.evidence-panel {
  margin: 0;
  padding: 0;
}
.evidence-panel summary {
  list-style: none;
  cursor: pointer;
  padding: 14px 16px;
  color: #d5dce8;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .1em;
  font-size: 12px;
}
.evidence-panel summary::-webkit-details-marker { display: none; }
.evidence-panel[open] {
  padding-bottom: 14px;
}
.evidence-actions,
.evidence-list,
#jobLogMirror {
  margin: 0 16px;
}
.evidence-list {
  display: grid;
  gap: 10px;
  color: var(--muted);
}
.source-card {
  padding: 12px;
  border-radius: 14px;
  background: rgba(148, 163, 184, .09);
  border: 1px solid rgba(148, 163, 184, .16);
}
.source-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 8px 0;
}
.source-metrics span {
  border: 1px solid rgba(148, 163, 184, .18);
  border-radius: 999px;
  padding: 4px 8px;
  color: #d8e2f0;
  background: rgba(7, 10, 16, .42);
  font-size: 12px;
}
.source-card a { color: #93c5fd; text-decoration: none; }
@media (max-width: 980px) {
  .app-shell { grid-template-columns: 1fr; }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
}
"""


APP_JS = r"""
let selectedFile = "";
let selectedVideo = null;
let selectedChatKey = "";
let currentJob = "";
let pollTimer = null;
let lastEvidenceQuestion = "";
let lastEvidenceDataset = "";
const CHAT_STORAGE_KEY = "youtube-rag-chat-histories-v1";
const answerCache = new Map();
const chatHistories = loadChatHistories();

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<strong>$1</strong>");
}

function formatAnswer(value) {
  const lines = String(value || "").split(/\r?\n/);
  const parts = [];
  let listItems = [];

  function flushList() {
    if (!listItems.length) return;
    parts.push("<ul>" + listItems.map(item => `<li>${formatInlineMarkdown(item)}</li>`).join("") + "</ul>");
    listItems = [];
  }

  for (const line of lines) {
    const trimmed = line.trim();
    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      listItems.push(bullet[1]);
      continue;
    }
    flushList();
    if (trimmed) parts.push(`<p>${formatInlineMarkdown(trimmed)}</p>`);
  }
  flushList();
  return parts.join("");
}

function loadChatHistories() {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY);
    const data = raw ? JSON.parse(raw) : {};
    return new Map(Object.entries(data));
  } catch {
    return new Map();
  }
}

function persistChatHistories() {
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(Object.fromEntries(chatHistories)));
  } catch {
    // Chat remains available in memory if browser storage is blocked.
  }
}

function chatKeys(file = selectedFile, video = selectedVideo) {
  return [...new Set([
    file,
    video?.name,
    video?.url,
    video?.title
  ].filter(Boolean))];
}

function datasetChatKey(video) {
  if (!video) return "";
  return video.url || video.name || video.file || video.title || "";
}

function getChatHistory(file = selectedFile, video = selectedVideo) {
  for (const key of chatKeys(file, video)) {
    const history = chatHistories.get(key);
    if (history && history.length) return history;
  }
  return [];
}

function setChatHistory(file, video, history) {
  for (const key of chatKeys(file, video)) {
    chatHistories.set(key, history);
  }
  persistChatHistories();
}

function rememberMessage(role, content) {
  rememberMessageFor(selectedChatKey || selectedFile, selectedVideo, role, content);
}

function rememberMessageFor(file, video, role, content) {
  if (!file) return;
  const history = [...getChatHistory(file, video)];
  history.push({ role, content });
  setChatHistory(file, video, history);
}

function snapshotCurrentMessages() {
  const key = selectedChatKey || selectedFile;
  if (!key) return;
  const nodes = [...document.querySelectorAll("#messages .message")];
  const history = nodes.map(node => {
    const role = node.classList.contains("user") ? "user" : "assistant";
    const content = node.querySelector(".bubble")?.innerText?.trim() || "";
    return { role, content };
  }).filter(item => item.content);

  const hasUserMessage = history.some(item => item.role === "user");
  if (!hasUserMessage) return;

  setChatHistory(key, selectedVideo, history);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

function addMessage(role, content, persist = true) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `<div class="bubble">${role === "assistant" ? formatAnswer(content) : escapeHtml(content)}</div>`;
  $("messages").appendChild(div);
  $("messages").scrollTop = $("messages").scrollHeight;
  if (persist) rememberMessage(role, content);
  return div.querySelector(".bubble");
}

function renderMessages() {
  const key = selectedChatKey || selectedFile;
  const history = key ? getChatHistory(key, selectedVideo) : [];
  if (history.length) {
    $("messages").innerHTML = "";
    history.forEach(item => addMessage(item.role, item.content, false));
    return;
  }

  $("messages").innerHTML = `
    <div class="message assistant">
      <div class="bubble">Video đã sẵn sàng. Bạn có thể hỏi về nội dung, các bước thực hành, hoặc mở phần kiểm chứng để xem timestamp.</div>
    </div>
  `;
}

function renderHero(video) {
  if (!video) {
    $("videoHero").className = "video-hero";
    $("videoHero").style.backgroundImage = "";
    $("videoHero").innerHTML = `
      <div class="empty-state">
        <h2>Chọn một video hoặc gửi link mới</h2>
        <p>App sẽ crawl transcript, build vector index, lưu thành dataset, rồi mở một khung hỏi đáp có nguồn timestamp.</p>
      </div>
    `;
    return;
  }

  $("videoHero").className = "video-hero" + (video.thumbnail ? " has-thumb" : "");
  $("videoHero").style.backgroundImage = video.thumbnail
    ? `linear-gradient(90deg, rgba(8, 11, 18, .96), rgba(8, 11, 18, .5)), url("${video.thumbnail}")`
    : "";
  $("videoHero").innerHTML = `
    <div class="video-title">
      <h2>${escapeHtml(video.title)}</h2>
      <p>${escapeHtml(video.channel || "YouTube")} · ${escapeHtml(video.duration || "-")} · ${escapeHtml(video.name || "")}</p>
    </div>
  `;
}

function renderVideos(items) {
  const root = $("videoList");
  root.innerHTML = "";
  if (!items.length) {
    root.innerHTML = `<div class="video-card"><strong>Chưa có video nào</strong><small>Hãy gửi một link YouTube để bắt đầu.</small></div>`;
    return;
  }
  items.forEach(item => {
    const div = document.createElement("div");
    div.className = "video-card" + (item.file === selectedFile ? " active" : "");
    div.innerHTML = `
      <strong>${escapeHtml(item.title)}</strong>
      <small>${escapeHtml(item.name)}</small>
      <span class="ready-pill">${item.has_index ? "Sẵn sàng hỏi đáp" : "Đang thiếu index"}</span>
    `;
    div.onclick = () => {
      snapshotCurrentMessages();
      selectedFile = item.file;
      selectedVideo = item;
      selectedChatKey = datasetChatKey(item);
      lastEvidenceQuestion = "";
      lastEvidenceDataset = "";
      renderVideos(items);
      renderHero(item);
      renderMessages();
      $("questionInput").disabled = !item.has_index;
      $("askButton").disabled = !item.has_index;
      $("evidenceList").textContent = "Chưa có kết quả truy xuất.";
    };
    root.appendChild(div);
  });
}

async function refreshVideos() {
  snapshotCurrentMessages();
  const data = await api("/api/datasets");
  renderVideos(data.datasets);
  if (selectedFile) {
    const match = data.datasets.find(item => item.file === selectedFile);
    if (match) {
      selectedVideo = match;
      selectedChatKey = datasetChatKey(match);
      renderHero(match);
      renderMessages();
      $("questionInput").disabled = !match.has_index;
      $("askButton").disabled = !match.has_index;
    }
  }
}

function inferProgress(logText, status) {
  if (status === "completed") return 100;
  if (status === "failed") return 100;
  if (/Building FAISS|Saved FAISS/i.test(logText)) return 92;
  if (/Generating embeddings|Batches/i.test(logText)) return 78;
  if (/Generating RAG|Saved to/i.test(logText)) return 62;
  if (/Fetching transcripts|transcript/i.test(logText)) return 42;
  if (/metadata|Found|Tìm thấy/i.test(logText)) return 25;
  return 12;
}

function pollJob() {
  if (!currentJob) return;
  api("/api/jobs/" + currentJob).then(job => {
    const logText = job.log.length ? job.log.join("\n") : "Đang khởi động pipeline...";
    const progress = inferProgress(logText, job.status);
    $("jobPanel").hidden = false;
    $("jobStage").textContent = job.status === "running" ? "Processing" : job.status;
    $("jobProgressText").textContent = `${progress}%`;
    $("jobProgress").style.width = `${progress}%`;
    $("jobLog").textContent = logText;
    $("jobLogMirror").textContent = logText;
    $("jobLog").scrollTop = $("jobLog").scrollHeight;

    if (job.status !== "running") {
      clearInterval(pollTimer);
      selectedFile = job.output_file;
      refreshVideos();
      if (job.status === "completed") {
        $("jobPanel").hidden = true;
      }
    }
  }).catch(err => {
    $("jobLog").textContent = err.message;
    $("jobLogMirror").textContent = err.message;
  });
}

$("ingestForm").onsubmit = async (event) => {
  event.preventDefault();
  const videoUrl = $("videoUrl").value.trim();
  if (!videoUrl) return;
  $("jobPanel").hidden = false;
  $("jobStage").textContent = "Queued";
  $("jobProgressText").textContent = "0%";
  $("jobProgress").style.width = "0%";
  $("jobLog").textContent = "Đang tạo phiên xử lý...";
  $("jobLogMirror").textContent = "Đang tạo phiên xử lý...";
  const job = await api("/api/jobs", {
    method: "POST",
    body: JSON.stringify({
      video_url: videoUrl,
      output_name: "",
      delay: 15,
      transcribe_missing: $("transcribeMissing").checked
    })
  });
  currentJob = job.id;
  clearInterval(pollTimer);
  pollTimer = setInterval(pollJob, 1500);
  pollJob();
};

$("askForm").onsubmit = async (event) => {
  event.preventDefault();
  if (!selectedFile) return;
  const question = $("questionInput").value.trim();
  if (!question) return;
  const activeFile = selectedFile;
  const activeVideo = selectedVideo;
  const activeChatKey = selectedChatKey || selectedFile;
  addMessage("user", question);
  $("questionInput").value = "";
  const cacheKey = `${activeFile}::${question.toLowerCase()}`;
  if (answerCache.has(cacheKey)) {
    addMessage("assistant", answerCache.get(cacheKey));
    return;
  }
  const bubble = addMessage("assistant", "Đang trả lời...", false);
  try {
    const data = await api("/api/ask", {
      method: "POST",
      body: JSON.stringify({ output_file: activeFile, question, use_llm: true })
    });
    const answer = data.answer || "Không có kết quả.";
    answerCache.set(cacheKey, answer);
    bubble.innerHTML = formatAnswer(answer);
    rememberMessageFor(activeChatKey, activeVideo, "assistant", answer);
  } catch (err) {
    const message = err.message || "Không thể tạo câu trả lời.";
    bubble.innerHTML = formatAnswer(message);
    rememberMessageFor(activeChatKey, activeVideo, "assistant", message);
  }
};

async function updateEvidence() {
  if (!selectedFile) return;
  const question = $("questionInput").value.trim()
    || [...document.querySelectorAll(".message.user .bubble")].at(-1)?.textContent?.trim()
    || "";
  if (!question) return;
  if (question === lastEvidenceQuestion && selectedFile === lastEvidenceDataset) return;

  $("evidenceList").textContent = "Đang tìm đoạn liên quan...";
  const data = await api("/api/ask", {
    method: "POST",
    body: JSON.stringify({ output_file: selectedFile, question, use_llm: false })
  });
  lastEvidenceQuestion = question;
  lastEvidenceDataset = selectedFile;
  if (!data.sources || !data.sources.length) {
    $("evidenceList").textContent = "Chưa tìm thấy đoạn liên quan.";
    return;
  }
  $("evidenceList").innerHTML = data.sources.map(src => `
    <div class="source-card">
      <strong>Chunk #${escapeHtml(src.rank)} · ${escapeHtml(src.time || "")}</strong>
      <div class="source-metrics">
        <span>Similarity: ${formatScore(src.score)}</span>
        <span>Rerank: ${formatScore(src.rerank_score)}</span>
      </div>
      <p>${escapeHtml(src.text || "")}</p>
      <a href="${escapeHtml(src.url || "")}" target="_blank">Mở timestamp</a>
    </div>
  `).join("");
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(4);
}

$("evidenceButton").onclick = updateEvidence;
$("evidencePanel").addEventListener("toggle", () => {
  if ($("evidencePanel").open) updateEvidence();
});
$("refreshVideos").onclick = refreshVideos;
window.addEventListener("beforeunload", snapshotCurrentMessages);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) snapshotCurrentMessages();
});

renderHero(null);
refreshVideos();
"""


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = NEW_INDEX_HTML.encode("utf-8")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, socket.timeout):
                pass
            return

        if parsed.path == "/static/app.css":
            body = APP_CSS.encode("utf-8")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/css; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, socket.timeout):
                pass
            return

        if parsed.path == "/static/app.js":
            body = APP_JS.encode("utf-8")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, socket.timeout):
                pass
            return

        if parsed.path == "/api/datasets":
            json_response(self, {"datasets": list_datasets()})
            return

        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                payload = dict(job) if job else None
            if not payload:
                json_response(self, {"error": "Job not found"}, 404)
                return
            json_response(self, payload)
            return

        json_response(self, {"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/jobs":
                payload = read_json_body(self)
                video_url = (payload.get("video_url") or "").strip()
                if not video_url:
                    json_response(self, {"error": "Missing video_url"}, 400)
                    return
                job = create_job(
                    video_url=video_url,
                    output_name=payload.get("output_name") or "",
                    delay=int(payload.get("delay") or 15),
                    transcribe_missing=bool(payload.get("transcribe_missing")),
                )
                json_response(self, job, 201)
                return

            if parsed.path == "/api/ask":
                payload = read_json_body(self)
                output_file = payload.get("output_file") or ""
                question = (payload.get("question") or "").strip()
                use_llm = bool(payload.get("use_llm"))
                if not output_file or not question:
                    json_response(self, {"error": "Missing output_file or question"}, 400)
                    return
                if not Path(output_file).exists():
                    json_response(self, {"error": "Dataset not found"}, 404)
                    return
                json_response(self, answer_question(output_file, question, use_llm))
                return

            json_response(self, {"error": "Not found"}, 404)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, 500)

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser(description="Local web UI for YouTube RAG.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Web app running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
