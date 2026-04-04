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
| **Stream** | Real-time live stream / webcam processing with adaptive segmentation and two-level memory |
| **Resilience** | Retry with exponential backoff, graceful degradation for optional skills, lifecycle hooks |

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

**Qwen3.5 (recommended):**
```bash
# Requires vLLM nightly for Qwen3.5 support
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

# Auto-detect local model or download from HuggingFace
bash scripts/serving_qwen3_5.sh

# Or manually:
vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --allowed-local-media-path $(pwd)/cache
```

**Qwen3-VL (legacy):**
```bash
bash scripts/serving_qwen3vl.sh
```

On a GPU cluster:
```bash
TP_SIZE=2 MAX_MODEL_LEN=131072 bash scripts/serving_qwen3_5.sh
```

### 3. Run

**CLI:**
```bash
python agent/main.py analyze youtube "https://www.youtube.com/watch?v=..." --mode detailed

# With structured JSON logging
python agent/main.py --log-format json analyze youtube "https://www.youtube.com/watch?v=..." --mode detailed
```

**REST API:**
```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000

# Standard (returns final JSON)
curl -X POST http://localhost:9000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"youtube", "uri":"https://www.youtube.com/watch?v=...", "mode":"detailed"}'

# Streaming (returns Server-Sent Events with real-time progress)
curl -N -X POST http://localhost:9000/analyze/stream \
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

# Live stream from webcam
python agent/main.py local webcam --mode live

# Live stream from RTMP/HTTP URL
python agent/main.py local stream --mode live --stream-source stream --stream-url rtmp://host/live/key
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

## Online / Streaming Processing

VidCopilot supports real-time video understanding from webcams and RTMP/HTTP streams. The streaming architecture is inspired by [InternLM-XComposer-2.5-OmniLive](https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-2.5-OmniLive).

### Architecture

The streaming pipeline uses a three-module design:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Live Stream Pipeline                        │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐   │
│  │  Perception   │   │     Memory       │   │    Reasoning      │   │
│  │              │   │                  │   │                   │   │
│  │ Frame capture│──▶│ Local segments   │──▶│ Query retrieval   │   │
│  │ Scene detect │   │ Global summary   │   │ LLM Q&A          │   │
│  │ SlowFast     │   │ Backup-on-query  │   │ Context building  │   │
│  └──────────────┘   └──────────────────┘   └───────────────────┘   │
│       │                                            │               │
│       ▼                                            ▼               │
│  Frame-level results                     Answer + evidence         │
└─────────────────────────────────────────────────────────────────────┘
```

**Module A: Perception** — Captures frames at configurable FPS, detects scene changes using CLIP embedding similarity (threshold-based, not fixed windows), and routes each frame through heavy or light analysis models (SlowFast strategy).

**Module B: Memory** — Maintains a two-level memory hierarchy:
  - *Local memory*: Per-segment compressed representations (caption + CLIP embedding) for fine-grained temporal retrieval
  - *Global memory*: LLM-generated summary across all segments for holistic understanding

**Module C: Reasoning** — On query, snapshots the memory (backup-on-query pattern for consistency), retrieves relevant segments via cosine similarity, and generates answers using the full context.

### Key Features

| Feature | Description |
|---------|-------------|
| **Adaptive segmentation** | CLIP-based scene-change detection creates semantically meaningful segments instead of fixed-duration windows |
| **SlowFast analysis** | Heavy model (7B MLLM + OCR + detection) every N frames; light model (small MLLM + OCR) on others |
| **Two-level memory** | Local per-segment memory for retrieval + global summary for holistic understanding |
| **Live Q&A** | Ask questions mid-stream; memory is snapshotted for consistency while processing continues |
| **Backup-on-query** | Deep-copy of memory state ensures retrieval operates on consistent data |

### CLI Usage

```bash
# Live stream from webcam (default)
python agent/main.py analyze local webcam --mode live

# RTMP stream
python agent/main.py analyze local stream --mode live \
  --stream-source stream --stream-url rtmp://host/live/key

# HTTP stream (e.g., IP camera)
python agent/main.py analyze local stream --mode live \
  --stream-source stream --stream-url http://camera-ip/video
```

### REST API

Start the server, then use the `/live/*` endpoints:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000
```

**Start a session:**
```bash
curl -X POST http://localhost:9000/live/start \
  -H 'Content-Type: application/json' \
  -d '{"source": "stream", "stream_url": "rtmp://host/live/key", "fps": 1}'
# Returns: {"session_id": "live_0001", "status": "started"}
```

**Ask a question mid-stream:**
```bash
curl -X POST http://localhost:9000/live/ask \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "live_0001", "question": "What is happening in the video?"}'
# Returns: {"answer": "...", "relevant_segments": [...], "global_summary": "..."}
```

**Check status:**
```bash
curl http://localhost:9000/live/status/live_0001
# Returns: {"running": true, "segments_processed": 12, "total_duration_sec": 180.0, ...}
```

**Stop and get final memory:**
```bash
curl -X POST http://localhost:9000/live/stop \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "live_0001"}'
# Returns: {"memory": {...}, "total_frame_results": 180}
```

### Streaming Configuration

Settings in `workflows.yaml` under `live_stream`:

```yaml
live_stream:
  source: webcam               # "webcam" or "stream"
  fps: 1                       # frames per second to process
  heavy_interval: 5            # use heavy model every N frames
  similarity_threshold: 0.9    # CLIP cosine similarity for scene change
  min_segment_frames: 3        # minimum frames before allowing new segment
  max_segment_frames: 16       # force new segment after this many frames
```

Model tiers in `models.yaml`:

```yaml
mllm:
  heavy:
    model_name: qwen3.5-9b     # Qwen3.5 unified VL model (recommended)
    base_url: http://localhost:8000/v1
  light:
    model_name: qwen3.5-4b     # Lightweight model for fast per-frame captioning
    base_url: http://localhost:8000/v1
```

## E2E Testing

The `test_youtube_e2e.py` script auto-discovers or launches model serving, downloads a YouTube video, and runs a full test suite:

```bash
# Auto-detect/launch serving + run all tests
python scripts/test_youtube_e2e.py

# Use existing endpoint
python scripts/test_youtube_e2e.py --api-base http://localhost:8000/v1

# Custom video, specific tests
python scripts/test_youtube_e2e.py \
    --youtube "https://www.youtube.com/watch?v=..." \
    --tests frames qa multi_turn_qa
```

Tests: `frames` | `batch_frames` | `video_caption` | `qa` | `multi_turn_qa`

See [Testing Guide](docs/testing.md) for full details.

## Production Features

VidCopilot includes production-hardening patterns inspired by large-scale agent architectures:

### Retry with Exponential Backoff

All model calls (vLLM chat, Whisper ASR, embedding API) are wrapped with automatic retry on transient failures (timeouts, connection errors, 5xx, rate limits). Configurable per-call: `max_retries`, `base_delay`, `max_delay` with jitter to avoid thundering herd.

```python
from agent.core.retry import retry_with_backoff

@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0)
def my_api_call():
    ...
```

### Graceful Degradation

Optional skills (OCR, object detection, emotion analysis, translation, web search) are wrapped with `@skill_guard` — if a dependency is missing or a model fails, the skill is skipped and the pipeline continues with a warning instead of crashing.

### Parallel Skill Execution

In the `detailed` workflow, independent skills (OCR, object detection, emotion analysis) run in parallel using a thread pool. Configurable via `max_parallel_skills` in `workflows.yaml` (default: 3).

### Streaming Progress Events

An event bus (`agent.core.events`) emits lifecycle events (`skill_start`, `skill_complete`, `skill_error`, `skill_skipped`, `progress`) at each pipeline step.

- **CLI**: Real-time per-skill progress printed to stderr
- **API**: `POST /analyze/stream` returns Server-Sent Events for live progress monitoring

### Lifecycle Hooks

Shell commands can be triggered at analysis milestones via `hooks.yaml`:

```yaml
hooks:
  post_analysis:
    - command: "curl -X POST $WEBHOOK_URL -d @$RESULT_PATH"
      async: true
      timeout: 10
  on_error:
    - command: "echo 'Failed: $ERROR_MSG' >> errors.log"
```

Hook points: `pre_analysis`, `post_analysis`, `post_skill`, `on_error`, `post_highlight`, `post_index`.

### Structured Logging

Pass `--log-format json` to the CLI for machine-readable JSON logs with `video_id`, `skill_name`, `duration_ms`, and `status` fields. Use `WorkflowTracker` for per-workflow skill timing summaries.

## Project Structure

```
agent/
  core/
    schemas.py           # Data models (VideoAsset, FrameSet, Transcript, ContentMetadata, ...)
    orchestrator.py      # Workflow router with hook triggers
    retry.py             # @retry_with_backoff decorator (exponential backoff + jitter)
    skill_guard.py       # @skill_guard decorator (graceful degradation)
    events.py            # EventBus for streaming progress notifications
    parallel.py          # Parallel skill execution via ThreadPoolExecutor
    hooks.py             # Lifecycle hook manager (reads hooks.yaml)
    logging_config.py    # Structured JSON logging, WorkflowTracker
  extensions/
    models/              # vLLM client, direct model loader
    skills/              # Processing skills
      subtitle_parser.py   # VTT/SRT parsing into Transcript
      content_sufficiency.py # Heuristic check: skip visuals if transcript is enough
      video_download.py    # yt-dlp with metadata + subtitle extraction
      asr.py               # Whisper ASR
      vision_caption.py    # MLLM frame/video captioning
      timeline_builder.py  # LLM-based timeline (uses content metadata)
      scene_similarity.py  # CLIP-based scene-change detection for streaming
      stream_memory.py     # Two-level memory manager (local + global)
      live_stream_processing.py # Real-time stream processor with SlowFast
      ...                  # OCR, object detection, emotion, FAISS, etc.
    workflows/           # Pipelines (brief, detailed, index, ask, highlights, report, live)
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
- `hooks.yaml` — lifecycle hooks (shell commands triggered at analysis milestones)

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
- **Models:** Qwen3.5 (default, recommended), Qwen3-VL (legacy), configurable via `models.yaml`
- **vLLM:** Nightly build required for Qwen3.5 (`uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly`)

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
