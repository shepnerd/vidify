# VidCopilot

Video understanding agent — feed it a YouTube URL and get structured analysis, searchable index, Q&A, highlights, and reports.

## What It Does

| Capability | Description |
|------------|-------------|
| **Analyze** | Download, extract subtitles/metadata, ASR, conditionally caption frames, build timeline |
| **Understand** | OCR, object detection, emotion analysis, translation |
| **Search** | FAISS index over frames + ASR + metadata, semantic Q&A with targeted visual lookup |
| **Edit** | Auto-detect highlights, export clips, assemble reels |
| **Enhance** | Web search context, multi-language support |
| **Report** | Comprehensive analysis report generation |

## Design Philosophy: ASR-First, Visuals as Last Resort

Most videos (documentaries, vlogs, presentations, interviews, movie reviews, sports commentary) convey their key information through speech or subtitles. VidCopilot is designed around this insight:

1. **Subtitles first** — For YouTube/web videos, embedded subtitles (manual or auto-generated) are extracted via yt-dlp and used as the primary transcript. These are free and often higher quality than ASR.
2. **ASR fallback** — If no subtitles are available, Whisper ASR transcribes the audio.
3. **Metadata context** — Video title, description, tags, and uploader info from the source platform are extracted and used as context for timeline building and Q&A.
4. **Sufficiency check** — A fast heuristic (no LLM call) assesses whether the transcript covers enough of the video to skip expensive MLLM visual processing. Criteria: speech coverage ratio (default ≥30%) and word count (default ≥50 words).
5. **Conditional visual processing** — MLLM frame captioning only runs when the transcript is insufficient (e.g., silent videos, music videos, or videos with minimal speech).
6. **Targeted visual lookup** — In Q&A mode, when a question requires visual details (e.g., "what equation is on the board?"), only the frames at relevant timestamps are sampled and captioned, rather than the entire video.

This approach makes video understanding both smarter and faster — a 30-minute lecture with subtitles can be analyzed without a single MLLM call.

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
# System deps: ffmpeg, yt-dlp, Python 3.11+
```

### 2. Start model serving

```bash
vllm serve /path/to/qwen-vl \
  --host 0.0.0.0 --port 8000 \
  --allowed-local-media-path $(pwd)/cache
```

On a GPU cluster:
```bash
bash scripts/serving_qwen3vl.sh
```

### 3. Run

**CLI:**
```bash
python agent/main.py youtube "https://www.youtube.com/watch?v=..." --mode detailed
```

**REST API:**
```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000
curl -X POST http://localhost:9000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"youtube", "uri":"https://www.youtube.com/watch?v=...", "mode":"detailed"}'
```

**Web GUI:**
Open `http://localhost:9000` after starting the server.

## Workflow Modes

```bash
# Quick summary (ASR-first, skips visuals if transcript is sufficient)
python agent/main.py youtube URL --mode brief

# Full analysis (OCR, emotions, objects, ASR, translation)
python agent/main.py youtube URL --mode detailed

# Force visual processing even when transcript is sufficient
python agent/main.py youtube URL --mode brief --force-visual

# Build search index, then ask questions
python agent/main.py youtube URL --mode ask --question "What are the key conclusions?"

# Ask a visual question (triggers targeted frame lookup)
python agent/main.py youtube URL --mode ask --question "What equation is shown on the board at 5:30?"

# Export highlight clips
python agent/main.py youtube URL --mode highlights

# Generate report with web search
python agent/main.py youtube URL --mode report --include-web-search
```

### Processing Flow

```
                    ┌─────────────┐
                    │  Download   │
                    │  + Metadata │ ← yt-dlp extracts info.json, subtitles
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Probe     │ ← ffprobe: duration, fps, resolution
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │  Subtitles available?    │
              └──┬──────────────────┬───┘
                 │ yes              │ no
          ┌──────▼──────┐   ┌──────▼──────┐
          │ Parse subs  │   │ Whisper ASR │
          └──────┬──────┘   └──────┬──────┘
                 └────────┬────────┘
                          │
                   ┌──────▼──────┐
                   │ Sufficiency │ ← coverage ≥ 30%? words ≥ 50?
                   │   check     │
                   └──┬──────┬───┘
                      │      │
            sufficient│      │ insufficient
                      │      │
               ┌──────▼──┐ ┌─▼──────────┐
               │  Skip   │ │ MLLM frame │
               │  MLLM   │ │ captioning │
               └──────┬──┘ └─┬──────────┘
                      └───┬──┘
                          │
                   ┌──────▼──────┐
                   │  Timeline   │ ← uses transcript + metadata + frames (if any)
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │    Save     │
                   └─────────────┘
```

## E2E Testing

The `test_youtube_e2e.py` script auto-discovers or launches model serving, downloads a YouTube video, and runs a full test suite:

```bash
# Auto-detect/launch serving + run all tests
python scripts/test_youtube_e2e.py

# Use existing endpoint
python scripts/test_youtube_e2e.py --api-base http://10.0.0.5:8000/v1

# Custom video, specific tests
python scripts/test_youtube_e2e.py \
    --youtube "https://www.youtube.com/watch?v=..." \
    --tests frames qa multi_turn_qa
```

Tests: `frames` | `batch_frames` | `video_caption` | `qa` | `multi_turn_qa`

See [Testing Guide](docs/testing.md) for full details.

## Project Structure

```
agent/
  core/
    schemas.py           # Data models (VideoAsset, FrameSet, Transcript, ContentMetadata, ...)
    orchestrator.py      # Workflow router
  extensions/
    models/              # vLLM client, direct model loader
    skills/              # Processing skills
      subtitle_parser.py   # VTT/SRT parsing into Transcript
      content_sufficiency.py # Heuristic check: skip visuals if transcript is enough
      video_download.py    # yt-dlp with metadata + subtitle extraction
      asr.py               # Whisper ASR
      vision_caption.py    # MLLM frame/video captioning
      timeline_builder.py  # LLM-based timeline (uses content metadata)
      ...                  # OCR, object detection, emotion, FAISS, etc.
    workflows/           # Pipelines (brief, detailed, index, ask, highlights, report)
    utils/               # Caching, hashing
  config.py              # YAML config loader
  main.py                # CLI entry point
server/
  app.py                 # FastAPI REST server (port 9000)
scripts/                 # Test and demo scripts
docs/                    # Detailed documentation
```

## Configuration

Optional YAML files in the project root:

- `models.yaml` — model selection, parameters, endpoints
- `workflows.yaml` — workflow steps, frame limits, feature toggles

### ASR-First Configuration

These settings in `workflows.yaml` control the ASR-first behavior:

```yaml
brief:
  asr_first: true                    # enable ASR-first mode (default: true)
  min_coverage_ratio: 0.3            # minimum speech-to-video duration ratio
  min_word_count: 50                 # minimum transcript words to be "sufficient"
  force_visual: false                # override: always run MLLM captioning
  prefer_subtitles_over_asr: true    # use embedded subs over Whisper when available
```

Falls back to built-in defaults if files don't exist. CLI/API parameters always take priority.

See [Configuration Guide](docs/configuration.md).

## Docker

```bash
# Full stack (app + vLLM)
docker-compose up

# App only
docker build -t vidcopilot .
docker run -p 9000:9000 vidcopilot
```

## Requirements

- **System:** ffmpeg, yt-dlp, Python 3.11+
- **GPU:** vLLM-compatible GPU for model serving (or use `--direct-model` for local loading)
- **Models:** Qwen3-VL (default), configurable via `models.yaml`

## Documentation

| Document | Contents |
|----------|----------|
| [Architecture](docs/architecture.md) | Data models, data flow, cache structure, model interfaces |
| [Workflows](docs/workflows.md) | All workflow pipelines in detail |
| [Skills Reference](docs/skills.md) | All skills — API signatures and descriptions |
| [API Reference](docs/api.md) | REST endpoints, CLI options, request/response schemas |
| [Configuration](docs/configuration.md) | YAML configs, vLLM setup, Docker deployment |
| [Testing Guide](docs/testing.md) | E2E test script, demo scripts, individual tests |
| [Web Search](docs/web-search.md) | Google/Baidu search integration setup |
