# Vidify

[简体中文](readme_cn.md)

Vidify is a video understanding agent. Give it a YouTube URL, HTTP video URL, or
local video and get structured analysis, searchable indexes, Q&A, highlights,
reports, and live-stream understanding.

## What It Does

| Capability | Description |
|------------|-------------|
| Analyze | Download media, extract subtitles/metadata, run ASR when needed, and build timelines |
| Understand | Caption frames, read OCR text, detect objects, analyze emotion, and translate transcripts |
| Search & Ask | Build a FAISS index over transcript, frames, and metadata for evidence-backed Q&A |
| Edit | Detect highlights, export clips, and optionally assemble reels |
| Stream | Process webcams or RTMP/HTTP streams with adaptive segmentation and live Q&A |
| Operate | Retry transient failures, degrade optional skills gracefully, emit progress events, and run hooks |

Vidify is ASR-first: subtitles and speech usually carry the main story, so visual
model calls are skipped when transcript coverage is sufficient. See
[Project Overview](docs/overview.md) for the full processing flow.

## Quick Start

### 1. Install

```bash
pip install -e .
```

System requirements: Python 3.11+, `ffmpeg`, and `yt-dlp`.

Optional feature groups:

```bash
pip install -e ".[asr,ocr,emotion,live,serving]"
pip install -r requirements-full.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` when you need custom model endpoints, model names, cache paths, or
web search credentials. Full details are in [Configuration](docs/configuration.md).

### 3. Start Model Serving

Vidify expects an OpenAI-compatible multimodal endpoint, usually vLLM:

```bash
# vLLM >= 0.19.0 is required for Qwen3.5 support.
pip install "vllm>=0.19.0"

bash scripts/serving_qwen3_5.sh
```

Manual example:

```bash
vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --allowed-local-media-path $(pwd)/cache
```

See [Deployment](docs/deployment.md) for GPU, Ascend/NPU, Docker, and validation
commands.

### 4. Run

CLI:

```bash
python -m agent.main analyze youtube "https://www.youtube.com/watch?v=..." --mode detailed
python -m agent.main analyze local media/example.mp4 --mode brief
python -m agent.main analyze local media/example.mp4 --mode ask --question "What changed?"
```

REST API and web UI:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000

curl -X POST http://localhost:9000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"youtube","uri":"https://www.youtube.com/watch?v=...","mode":"detailed"}'
```

Open `http://localhost:9000` for the web interface.

## Workflow Modes

`brief` is the canonical lightweight mode. `quick` is still accepted as a legacy
alias in the CLI and API.

| Mode | Use It For | Example |
|------|------------|---------|
| `brief` | Fast ASR-first summary | `python -m agent.main analyze youtube URL --mode brief` |
| `detailed` | OCR, object detection, emotion, translation, and richer timelines | `python -m agent.main analyze youtube URL --mode detailed` |
| `ask` | Question-answering over an indexed video | `python -m agent.main analyze youtube URL --mode ask --question "What are the conclusions?"` |
| `highlights` | Clip export and optional reels | `python -m agent.main analyze youtube URL --mode highlights` |
| `report` | Structured report generation, optionally with web search | `python -m agent.main analyze youtube URL --mode report --include-web-search` |
| `live` | Webcam, RTMP, or HTTP stream understanding | `python -m agent.main analyze local webcam --mode live` |

See [Workflows](docs/workflows.md) and [API Reference](docs/api.md) for complete
parameters and request schemas.

## Hermes

This repo ships a Hermes-native skill at `.agents/skills/media/vidify`.

```bash
python -m agent.main hermes install-skill
```

The installer symlinks the skill into `~/.hermes/skills/media/vidify` by default.
Use `--strategy copy` for a standalone copy. The legacy `openclaw/` skill remains
available for older setups.

## Testing

Run the fast test suite:

```bash
pytest tests/
```

Validate against an existing model endpoint:

```bash
bash scripts/run_test_gpu.sh --api-base http://localhost:8000/v1 --video media/my_video.mp4
python scripts/test_all.py --video-path media/my_video.mp4 --api-base http://localhost:8000/v1
```

See [Testing Guide](docs/testing.md) for focused tests, YouTube E2E validation,
and hardware-specific notes.

## Repository Layout

| Path | Purpose |
|------|---------|
| `agent/core/` | Orchestration, schemas, events, hooks, retries, segmenting, and parallel execution |
| `agent/extensions/skills/` | Reusable video, audio, retrieval, and analysis units |
| `agent/extensions/workflows/` | User-facing workflow composition |
| `agent/extensions/models/` | Model adapters and direct-loading helpers |
| `server/` | FastAPI app, SSE endpoints, and web routes |
| `templates/` | Web UI templates |
| `scripts/` | Serving, validation, and demo helpers |
| `docs/` | Architecture, workflow, deployment, and API documentation |
| `cache/` | Runtime artifacts; do not commit generated outputs |

## Documentation

| Document | Contents |
|----------|----------|
| [Project Overview](docs/overview.md) | ASR-first design, capability map, and processing flow |
| [Deployment](docs/deployment.md) | vLLM serving, GPU validation, Ascend/NPU helpers, and Docker |
| [Live Streaming](docs/live-streaming.md) | Webcam/stream architecture, CLI/API usage, and config |
| [Production Features](docs/production.md) | Retries, graceful degradation, parallelism, progress events, hooks, and logging |
| [Architecture](docs/architecture.md) | Data models, cache structure, model interfaces, and orchestrator |
| [Workflows](docs/workflows.md) | Brief, detailed, index, ask, highlights, report, and live modes |
| [Skills Reference](docs/skills.md) | Skill APIs and responsibilities |
| [API Reference](docs/api.md) | REST endpoints, CLI arguments, examples, and schemas |
| [Configuration](docs/configuration.md) | YAML files, environment variables, vLLM setup, and Docker |
| [Testing Guide](docs/testing.md) | Pytest, local E2E, GPU/Ascend endpoint validation, and YouTube E2E |
| [Web Search](docs/web-search.md) | Google Custom Search and fallback search setup |
