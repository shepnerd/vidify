# VidCopilot

Video understanding agent — feed it a YouTube URL and get structured analysis, searchable index, Q&A, highlights, and reports.

## What It Does

| Capability | Description |
|------------|-------------|
| **Analyze** | Download, decode, sample key frames, caption, ASR, build timeline |
| **Understand** | OCR, object detection, emotion analysis, translation |
| **Search** | FAISS index over frames + ASR + metadata, semantic Q&A |
| **Edit** | Auto-detect highlights, export clips, assemble reels |
| **Enhance** | Web search context, multi-language support |
| **Report** | Comprehensive analysis report generation |

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
# Quick summary
python agent/main.py youtube URL --mode brief

# Full analysis (OCR, emotions, objects, ASR, translation)
python agent/main.py youtube URL --mode detailed

# Build search index, then ask questions
python agent/main.py youtube URL --mode ask --question "What are the key conclusions?"

# Export highlight clips
python agent/main.py youtube URL --mode highlights

# Generate report with web search
python agent/main.py youtube URL --mode report --include-web-search
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
    schemas.py           # Data models (VideoAsset, FrameSet, Transcript, ...)
    orchestrator.py      # Workflow router
  extensions/
    models/              # vLLM client, direct model loader
    skills/              # 24 processing skills
    workflows/           # 7 pipelines (brief, detailed, index, ask, ...)
    utils/               # Caching, hashing
  config.py              # YAML config loader
  main.py                # CLI entry point
server/
  app.py                 # FastAPI REST server (port 9000)
scripts/                 # Test and demo scripts
docs/                    # Detailed documentation
```

## Documentation

| Document | Contents |
|----------|----------|
| [Architecture](docs/architecture.md) | Data models, data flow, cache structure, model interfaces |
| [Workflows](docs/workflows.md) | All 7 workflow pipelines in detail |
| [Skills Reference](docs/skills.md) | All 24 skills — API signatures and descriptions |
| [API Reference](docs/api.md) | REST endpoints, CLI options, request/response schemas |
| [Configuration](docs/configuration.md) | YAML configs, vLLM setup, Docker deployment |
| [Testing Guide](docs/testing.md) | E2E test script, demo scripts, individual tests |
| [Web Search](docs/web-search.md) | Google/Baidu search integration setup |

## Configuration

Optional YAML files in the project root:

- `models.yaml` — model selection, parameters, endpoints
- `workflows.yaml` — workflow steps, frame limits, feature toggles

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
