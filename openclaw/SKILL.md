---
name: vidify
description: Analyze videos end-to-end — frame captioning, ASR transcription, OCR, object detection, emotion analysis, timeline, highlights, and Q&A. Give it a YouTube URL, HTTP link, or local file.
homepage: https://github.com/user/vidify
metadata:
  { "openclaw": { "emoji": "🎥", "requires": { "bins": ["vidify", "ffmpeg"] }, "install": [ { "id": "pip", "kind": "node", "label": "Install Vidify (pip)", "bins": ["vidify"] } ] } }
---

# Vidify — Video Understanding Agent

Use Vidify when the user wants to deeply analyze a video: understand what happens, extract transcripts, find highlights, ask questions about content, detect objects/emotions, or generate a report.

## When to use

- "analyze this video" / "what happens in this video?"
- "transcribe this video" / "get the transcript"
- "find the highlights" / "create a highlight reel"
- "what does the speaker say about X?" (Q&A)
- "summarize this YouTube video in detail"
- "detect objects/emotions in this video"
- "generate a report for this video"

## Quick start (CLI)

Analyze a YouTube video (detailed mode — frames + ASR + OCR + objects + emotions + timeline):

```bash
vidify analyze youtube "https://www.youtube.com/watch?v=VIDEO_ID" --mode detailed
```

Quick analysis (frames + captions + timeline, no ASR):

```bash
vidify analyze youtube "https://www.youtube.com/watch?v=VIDEO_ID" --mode quick
```

Local file:

```bash
vidify analyze local "/path/to/video.mp4" --mode detailed
```

## Wrapper scripts

For convenience, use the wrapper scripts shipped with this skill:

### Analyze a video

```bash
{baseDir}/scripts/vidify-analyze.sh <source_type> "<uri>" [mode]
```

- `source_type`: `youtube`, `url`, or `local`
- `uri`: YouTube URL, HTTP URL, or local file path
- `mode`: `quick` or `detailed` (default: `detailed`)

### Ask a question about a video

```bash
{baseDir}/scripts/vidify-ask.sh <source_type> "<uri>" "<question>"
```

Automatically builds a FAISS index if needed, then answers the question with evidence and timestamps.

### Start the REST API server

```bash
{baseDir}/scripts/vidify-server.sh start
{baseDir}/scripts/vidify-server.sh stop
{baseDir}/scripts/vidify-server.sh status
```

Runs the Vidify API server on port 9000. Use when you need concurrent access or want to use the REST API directly.

## Workflow modes

| Mode | What it does |
|------|-------------|
| `quick` | Scene keyframes + vision captions + timeline |
| `detailed` | All of quick + ASR + OCR + object detection + emotion analysis + translation |
| `index` | Build FAISS semantic search index from analysis |
| `ask` | Answer questions about the video (needs index) |
| `highlights` | Detect and export highlight clips + optional reel |
| `report` | Generate a structured analysis report |

## CLI options

```
vidify analyze <source_type> <uri>
  --mode        quick|detailed|highlights|index|ask|report  (default: detailed)
  --cache-root  Cache directory                              (default: ./cache)
  --question    Question text (for ask mode)
  --max-frames  Max frames to sample                         (default: 128)
  --config      Config file path                             (default: config.yaml)
```

## REST API (when server is running)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze` | POST | Run quick/detailed analysis |
| `/index` | POST | Build FAISS index |
| `/ask` | POST | Ask a question (needs index) |
| `/highlights` | POST | Detect highlights |
| `/analysis` | POST | Load cached analysis |

Example API call:

```bash
curl -X POST http://localhost:9000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"youtube","uri":"https://www.youtube.com/watch?v=VIDEO_ID","mode":"detailed"}'
```

See `{baseDir}/references/api.md` for full request/response schemas.

## Prerequisites

- **Python 3.11+** with `vidify` installed (`pip install vidify` or `pip install -e /path/to/vidify`)
- **ffmpeg** on PATH (video/audio processing)
- **vLLM server** running on `http://localhost:8000/v1` with a multimodal model (e.g., Qwen3-VL) for captioning and Q&A
- **yt-dlp** on PATH (for YouTube downloads — installed as a Python dependency)

## Notes

- All results are cached under `./cache/videos/{hash}/`. Re-running the same video skips completed steps.
- The `ask` mode auto-builds the index if it doesn't exist yet. The index auto-runs `detailed` analysis if needed.
- Long videos may take significant time on first analysis. The cache makes subsequent queries fast.
- For best results, ensure the vLLM server is running with a capable vision-language model.
