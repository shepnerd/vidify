# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VidCopilot is a video understanding agent that takes a video source (YouTube URL, HTTP URL, or local file) and produces structured analysis: frame captioning, ASR transcription, OCR, object detection, emotion analysis, FAISS-based semantic search, Q&A, highlight detection, and report generation.

**Tech stack:** Python 3.11+, FastAPI, Pydantic v2, OpenAI SDK (targeting vLLM), Click CLI, FFmpeg, yt-dlp

## Commands

### Install
```bash
pip install -r requirements.txt
# or
python setup.py install
```

### Run CLI
```bash
python agent/main.py analyze <source_type> <uri> [--mode quick|detailed|highlights|index|ask|report] [--cache-root ./cache] [--max-frames 128]
# source_type: youtube, url, local
```

### Run API server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000
```

### Run tests
```bash
pytest tests/                          # Unit tests (CI uses this)
python scripts/test_youtube_e2e.py     # E2E test (requires vLLM + internet)
python scripts/test_youtube_e2e.py --api-base http://host:8000/v1 --tests frames qa
```

### Start vLLM model server
```bash
bash scripts/serving_qwen3vl.sh       # Launches vLLM on port 8000
```

### Docker
```bash
docker-compose up                      # Starts app (9000) + vLLM (8000)
```

## Architecture

### Data flow
```
CLI/API request → load_video() → run() orchestrator → wf_<mode>() workflow → skills → JSON output
```

### Key layers

- **`agent/core/schemas.py`** — Pydantic models: `VideoAsset`, `FrameSet`, `Transcript`, `TimelineChapter`, `HighlightClip`, `AnalysisResult`, etc. All data flows through these types.
- **`agent/core/orchestrator.py`** — Routes mode (brief/detailed/index/ask/highlights/report) to the corresponding workflow function.
- **`agent/extensions/workflows/`** — High-level pipelines that compose skills. `brief.py` and `detailed.py` are the primary analysis workflows; others (index, ask, highlights, report) build on their output.
- **`agent/extensions/skills/`** — 23 self-contained processing units (frame sampling, vision captioning, ASR, OCR, object detection, emotion analysis, FAISS indexing/search, etc.). Each skill is a standalone module with a main function.
- **`agent/extensions/models/`** — Model interface layer. `vllm_openai_client.py` wraps the OpenAI SDK to talk to vLLM; `direct_model_loader.py` loads models locally without a server.
- **`agent/config.py`** — Config loader with precedence: CLI params > YAML files (`models.yaml`, `workflows.yaml`) > built-in defaults. Default LLM endpoint is `http://localhost:8000/v1`.
- **`server/app.py`** — FastAPI REST API (endpoints: `/analyze`, `/index`, `/ask`, `/highlights`, `/report`, `/health`).

### Cache structure
Videos are cached under `cache/videos/{sha1(source_type:uri)}/` with subdirectories for frames, audio, analysis JSON, FAISS index, and highlight clips.

### Workflow dependencies
- `index` and `highlights` require a completed analysis (brief or detailed); they auto-run one if missing.
- `ask` requires an index; it auto-builds one if missing.

### Model interfaces
All LLM/embedding calls go through the OpenAI SDK pointed at a vLLM server (`/v1/chat/completions`, `/v1/embeddings`). Default model is Qwen3-VL (multimodal). Other models (Whisper, PaddleOCR, YOLOv8, Wav2Vec2) are loaded directly via their respective libraries.

## Configuration

- **`models.yaml`** — Model selection and parameters (MLLM, OCR, detection, ASR, emotion, translation)
- **`workflows.yaml`** — Workflow step definitions and feature toggles
- **`config.yaml`** — General config (merged with defaults from `agent/config.py`)
