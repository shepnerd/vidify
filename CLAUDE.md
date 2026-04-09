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
bash scripts/run_test_gpu.sh           # Full GPU test: launch vLLM + run 17-skill suite (reads .env)
bash scripts/run_test_gpu.sh --api-base http://host:8000/v1  # Reuse existing vLLM
python scripts/test_all.py --video-path media/video.mp4 --api-base http://host:8000/v1  # Manual
python scripts/test_youtube_e2e.py     # E2E test (requires vLLM + internet)
```

### Start vLLM model server
```bash
bash scripts/serving_qwen3_5.sh       # Qwen3.5-9B (recommended) — requires vLLM >= 0.19.0
bash scripts/serving_qwen3vl.sh       # Qwen3-VL (legacy) — launches vLLM on port 8000
```

### Docker
```bash
docker-compose up                      # Starts app (9000) + vLLM (8000)
```

### Ascend 910C / D-Cluster
```bash
# Submit vidcopilot job (2 NPUs, interactive)
job-run vidcopilot -f ./infra/d-cluster/job-vidcopilot.yaml

# Inside pod: start vLLM on NPU
bash scripts/serving_qwen3_5_ascend.sh

# One-command test on D-cluster (launches job, runs tests, cleans up)
bash scripts/run_test_ascend.sh --video media/video.mp4
bash scripts/run_test_ascend.sh --api-base http://10.x.x.x:8000/v1
```

**Image**: `registry2.d.pjlab.org.cn/ccr-a3-llmit/testenv:v0.1` — openEuler 24.03 aarch64, PyTorch 2.8 + torch_npu + CANN + vLLM 0.13.0 + vllm_ascend + all vidcopilot deps. **Do NOT reinstall** torch, torch_npu, transformers, or vllm inside this image.

**Infra scripts** (vcctl, kubectl wrappers) live in `../workbench/infra/d-cluster/`. Run `setup.sh` there for first-time setup.

**Cluster PyPI proxy**: `pip install -i https://pkg.pjlab.org.cn/repository/pypi-proxy/simple/ --trusted-host pkg.pjlab.org.cn`

## Architecture

### Data flow
```
CLI/API request → load_video() → run() orchestrator → wf_<mode>() workflow → skills → JSON output
```

### Key layers

- **`agent/core/schemas.py`** — Pydantic models: `VideoAsset`, `FrameSet`, `Transcript`, `TimelineChapter`, `HighlightClip`, `AnalysisResult`, etc. All data flows through these types.
- **`agent/core/orchestrator.py`** — Routes mode (brief/detailed/index/ask/highlights/report) to the corresponding workflow function.
- **`agent/core/segment.py`** — Parallel segment processing: `BaseSegmentor` ABC, `DurationSegmentor` (default, FFmpeg-based), `VideoSegment` model, result merge functions. Pluggable via `register_segmentor()` / `get_segmentor()` for future DL-based segmentors (e.g., TransNetV2, semantic boundary detection).
- **`agent/core/segment_worker.py`** — Per-segment processing worker: runs frame sampling → captioning → OCR/detection/emotion for one time slice, called from `run_segments_parallel()`.
- **`agent/core/parallel.py`** — `run_skills_parallel()` for concurrent skill execution within a segment; `run_segments_parallel()` for concurrent segment processing across a long video. Both use ThreadPoolExecutor with error isolation.
- **`agent/extensions/workflows/`** — High-level pipelines that compose skills. `brief.py` and `detailed.py` are the primary analysis workflows; others (index, ask, highlights, report) build on their output. Both `brief` and `detailed` support parallel segment processing for long videos (gated by config + duration threshold).
- **`agent/extensions/skills/`** — 23 self-contained processing units (frame sampling, vision captioning, ASR, OCR, object detection, emotion analysis, FAISS indexing/search, etc.). Each skill is a standalone module with a main function.
- **`agent/extensions/skills/frame_sampler.py`** — Supports `start_sec`/`end_sec` params for segment-level processing via FFmpeg `-ss`/`-to` flags (no physical video splitting).
- **`agent/extensions/models/`** — Model interface layer. `vllm_openai_client.py` wraps the OpenAI SDK to talk to vLLM; `direct_model_loader.py` loads models locally without a server.
- **`agent/config.py`** — Config loader with precedence: CLI params > YAML files (`models.yaml`, `workflows.yaml`) > built-in defaults. Default LLM endpoint is `http://localhost:8000/v1`.
- **`server/app.py`** — FastAPI REST API (endpoints: `/analyze`, `/index`, `/ask`, `/highlights`, `/report`, `/health`).

### Cache structure
Videos are cached under `cache/videos/{sha1(source_type:uri)}/` with subdirectories for frames, audio, analysis JSON, FAISS index, highlight clips, and parallel segment sub-caches (`segments/seg_000/`, etc.).

### Parallel segment processing
For long videos (configurable, default >5 min), the `detailed` and `brief` workflows split the video into temporal segments and process them concurrently:
- **Global steps** (sequential): probe, ASR/subtitles, sufficiency check, timeline builder, web search, translation
- **Per-segment steps** (parallel): frame sampling, MLLM captioning, OCR, object detection, emotion analysis
- Results are merged with timestamp adjustment before timeline generation
- Controlled by `parallel_segments` section in `workflows.yaml` (disabled by default)
- Segmentation strategy is pluggable via `BaseSegmentor` interface (`agent/core/segment.py`): default is `DurationSegmentor` (FFmpeg time ranges); future options include DL-based segmentors (TransNetV2, semantic boundary detection) via `register_segmentor()`

### Workflow dependencies
- `index` and `highlights` require a completed analysis (brief or detailed); they auto-run one if missing.
- `ask` requires an index; it auto-builds one if missing.

### Model interfaces
All LLM/embedding calls go through the OpenAI SDK pointed at a vLLM server (`/v1/chat/completions`, `/v1/embeddings`). Default model is Qwen3.5-9B (multimodal, unified VL). For Qwen3.5, thinking mode is disabled in pipeline calls via `enable_thinking: False` (see `agent/extensions/models/thinking.py`). Other models (Whisper, PaddleOCR, YOLOv8, Wav2Vec2) are loaded directly via their respective libraries.

## Configuration

- **`.env`** — Cluster config: `RL_CHARGED_GROUP`, `RL_MOUNT` (dual GPFS mounts), `CUDA_HOME` (for flashinfer JIT). Copy from `.env.example`.
- **`models.yaml`** — Model selection and parameters (MLLM, OCR, detection, ASR, emotion, translation)
- **`workflows.yaml`** — Workflow step definitions, feature toggles, and parallel segment config (`parallel_segments` section under `detailed`/`brief`)
- **`config.yaml`** — General config (merged with defaults from `agent/config.py`)

## GPU cluster usage

**Always `set -a && source .env && set +a` before launching GPU jobs via `scripts/rl.sh`.** The `.env` file defines `RL_CHARGED_GROUP=ptdata_gpu` which is required for GPU allocation. Without it, `rl.sh` falls back to `my_gpu_group` (default) and jobs fail with permission errors.

Example:
```bash
set -a && source .env && set +a
bash scripts/rl.sh -gpu 4 -- bash -c "vllm serve ..."
```

If start vllm for GPU jobs, remember to shutdown them down after finishing tasks.

### Qwen3-MLA model

Qwen3-MLA replaces GQA with Multi-head Latent Attention. **vLLM cannot serve this model** — use the transformers-based server instead:
```bash
bash scripts/serving_qwen3_mla.sh           # Launches transformers-based OpenAI-compatible server on port 8001
```
- Checkpoint: `/mnt/shared-storage-gpfs2/sfteval/xtuner_saved_model/internvl3.5/ablate_wuyue2/20260331093205/hf-5615`
- Custom modeling code: `modeling_qwen3_vl_mla.py` (loaded via `trust_remote_code=True`)
- MLA uses `kv_a_proj_with_mqa` + `kv_b_proj` instead of standard `q/k/v_proj` — incompatible with vLLM's built-in Qwen3VL handler

## Convensions
- Each time updates are introduced, remember to update README.md / CLAUDE.md and related setup files, if necessary. Trivial changes can be ignored for these files.