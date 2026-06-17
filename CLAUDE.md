# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vidify is a video understanding agent that takes a video source (YouTube URL, HTTP URL, or local file) and produces structured analysis: frame captioning, ASR transcription, OCR, object detection, emotion analysis, FAISS-based semantic search, Q&A, highlight detection, and report generation.

**Tech stack:** Python 3.11+, FastAPI, Pydantic v2, OpenAI SDK (targeting vLLM), Click CLI, FFmpeg, yt-dlp

## Commands

### Install
```bash
pip install -r requirements.txt
# or
python setup.py install

# MRA perception extras (needed for frame quality metrics, zoom crop, re-detection):
pip install vidify[mra]          # opencv-python, Pillow, ultralytics (YOLO)
# Or individually: pip install opencv-python Pillow ultralytics
# Note: MRA works without these (falls back to caption-based proxies), but
# real blur/brightness/contrast metrics and zoom_region/rerun_detector
# interventions require them.
```

### Run CLI
```bash
python -m agent.main analyze <source_type> <uri> [--mode brief|quick|detailed|highlights|index|ask|report|live|audit] [--cache-root ./cache] [--max-frames 128]
# source_type: youtube, url, local
python -m agent.main hermes install-skill

# MRA audit mode — runs base analysis then meta-reflective audit
python -m agent.main analyze local video.mp4 --mode audit
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
# Submit vidify job — Qwen3.5-9B with vLLM 0.18 (16 NPUs = full node, interactive)
job-run vidify-qwen35 -f ./infra/d-cluster/job-vidify-qwen35.yaml

# Inside pod: one-command start (downloads model, starts vLLM, launches chat)
bash scripts/start_vidify_ascend.sh /data/videos/myvideo.mp4

# Or start step by step:
bash scripts/serving_qwen3_5_ascend.sh &         # vLLM on NPU (Qwen3.5-9B, TP=4)
python -m agent.main chat local <video> --cache-root ./cache

# Legacy: Qwen2.5-VL-7B (for older images / fallback)
job-run vidify -f ./infra/d-cluster/job-vidify.yaml
bash scripts/serving_qwen2_5vl_ascend.sh &

# One-command test on D-cluster
bash scripts/run_test_ascend.sh --video media/video.mp4
bash scripts/run_test_ascend.sh --api-base http://10.x.x.x:8000/v1
```

**Images**:
- `registry2.d.pjlab.org.cn/ccr-hw/910c:vllm-ascend-0.18.0rc1-a3-0409` — vLLM 0.18.0 + vllm_ascend, openEuler 24.03 aarch64, PyTorch 2.9 + torch_npu 2.9 + CANN 8.5 + transformers 4.57. Supports both Qwen3.5-9B and Qwen2.5-VL-7B.

Upgrade transformers to >=4.51 inside the pod if using Qwen3.5 model type (already >=4.51 in current image). **Do NOT reinstall** torch, torch_npu, or vllm inside this image — pip will pull CPU-only versions that break NPU support.

**Model compatibility**:
- **Qwen3.5-9B** (recommended): Hybrid GDN+Attention, head_dim=256. Works with vllm_ascend >= 0.17 using `--enforce-eager`. num_attention_heads=16, num_kv_heads=4 → TP must be 1, 2, or 4.
- **Qwen2.5-VL-7B-Instruct** (fallback): head_dim=128, native NPU support. TP must divide 28 (use 2, 4, 7, or 14).

**NPU memory**: Use `--enforce-eager` (required for Qwen3.5 head_dim=256 and to avoid NPU graph capture OOM) and `--max-model-len 16384` (65536 causes OOM).

**Network**: Cluster nodes cannot reach the public internet. Use `HF_ENDPOINT=https://hf-mirror.com` for model downloads. PyPI proxy: `pip install -i https://pkg.pjlab.org.cn/repository/pypi-proxy/simple/ --trusted-host pkg.pjlab.org.cn`

**Infra scripts** (vcctl, kubectl wrappers) live in `../workbench/infra/d-cluster/`. Run `setup.sh` there for first-time setup.

## Architecture

### Data flow
```
CLI/API request → load_video() → run() orchestrator → wf_<mode>() workflow → skills → JSON output
```

### Key layers

- **`agent/core/schemas.py`** — Pydantic models: `VideoAsset`, `FrameSet`, `Transcript`, `TimelineChapter`, `HighlightClip`, `AnalysisResult`, etc. All data flows through these types.
- **`agent/core/orchestrator.py`** — Routes mode (brief/detailed/index/ask/highlights/report/audit) to the corresponding workflow function.
- **`agent/integrations/hermes.py`** — Stable Hermes-facing Python helpers plus a skill installer for `~/.hermes/skills`.
- **`agent/core/segment.py`** — Parallel segment processing: `BaseSegmentor` ABC, `DurationSegmentor` (default, FFmpeg-based), `VideoSegment` model, result merge functions. Pluggable via `register_segmentor()` / `get_segmentor()` for future DL-based segmentors (e.g., TransNetV2, semantic boundary detection).
- **`agent/core/segment_worker.py`** — Per-segment processing worker: runs frame sampling → captioning → OCR/detection/emotion for one time slice, called from `run_segments_parallel()`.
- **`agent/core/parallel.py`** — `run_skills_parallel()` for concurrent skill execution within a segment; `run_segments_parallel()` for concurrent segment processing across a long video. Both use ThreadPoolExecutor with error isolation.
- **`agent/extensions/workflows/`** — High-level pipelines that compose skills. `brief.py` and `detailed.py` are the primary analysis workflows; others (index, ask, highlights, report) build on their output. Both `brief` and `detailed` support parallel segment processing for long videos, and long-audio Whisper ASR split/merge (both gated by config + duration threshold).
- **`agent/extensions/skills/`** — 23 self-contained processing units (frame sampling, vision captioning, ASR, OCR, object detection, emotion analysis, FAISS indexing/search, etc.). Each skill is a standalone module with a main function.
- **`agent/extensions/skills/frame_sampler.py`** — Supports `start_sec`/`end_sec` params for segment-level processing via FFmpeg `-ss`/`-to` flags (no physical video splitting).
- **`agent/extensions/models/`** — Model interface layer. `vllm_openai_client.py` wraps the OpenAI SDK to talk to vLLM; `direct_model_loader.py` loads models locally without a server.
- **`agent/config.py`** — Config loader with precedence: CLI params > YAML files (`models.yaml`, `workflows.yaml`) > built-in defaults. Default LLM endpoint is `http://localhost:8000/v1`.
- **`server/app.py`** — FastAPI REST API (endpoints: `/analyze`, `/index`, `/ask`, `/highlights`, `/report`, `/health`).
- **`.agents/skills/media/vidify/`** — Hermes-native skill directory and wrappers Hermes can consume directly from the repo.
- **`agent/extensions/mra/`** — Meta-Reflective Auditor (MRA): a second-order audit module that validates the agent's own self-assessment. Runs a base analysis, generates a structured LLM reflection, scores it against evidence (rule-based), optionally executes targeted interventions (dense resample, zoom crop, re-detect, evidence-only re-reasoning), and produces a final accept/revise/abstain decision. Entry point: `runner.py:run_with_meta_reflection()`. See `design/meta_reflective_auditor_impl.md` for the full design.

### Cache structure
Videos are cached under `cache/videos/{sha1(source_type:uri)}/` with subdirectories for frames, audio, analysis JSON, FAISS index, highlight clips, and parallel segment sub-caches (`segments/seg_000/`, etc.). MRA writes `mra_audit.json` and intervention frames under `mra_frames/`.

### Parallel segment processing
For long videos (configurable, default >5 min), the `detailed` and `brief` workflows split the video into temporal segments and process them concurrently:
- **Global steps** (sequential): probe, ASR/subtitles, sufficiency check, timeline builder, web search, translation
- **Per-segment steps** (parallel): frame sampling, MLLM captioning, OCR, object detection, emotion analysis
- Results are merged with timestamp adjustment before timeline generation
- Controlled by `parallel_segments` section in `workflows.yaml` (disabled by default)
- Segmentation strategy is pluggable via `BaseSegmentor` interface (`agent/core/segment.py`): default is `DurationSegmentor` (FFmpeg time ranges); future options include DL-based segmentors (TransNetV2, semantic boundary detection) via `register_segmentor()`

Long audio can also use parallel ASR:
- `agent/extensions/skills/asr.py` supports clip-level split/merge with timestamp correction
- Controlled by `parallel_asr` under `brief` / `detailed` in `workflows.yaml`
- On Ascend helper scripts, the default is CPU workers (`VIDIFY_ASR_DEVICES=cpu,cpu,cpu,cpu`) so ASR does not contend with the 16-NPU vLLM pool unless you explicitly reserve accelerator devices
- Offline ASR accepts either `models/whisper-small` or `models/faster-whisper-small`

### Workflow dependencies
- `index` and `highlights` require a completed analysis (brief or detailed); they auto-run one if missing.
- `ask` requires an index; it auto-builds one if missing.
- `audit` runs a base analysis (brief by default, configurable via `workflows.yaml` → `audit.base_mode`), then layers the MRA loop on top. Does not require index.

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

If you start vLLM for GPU jobs, remember to shut it down after finishing tasks.

### Qwen3-MLA model

Qwen3-MLA replaces GQA with Multi-head Latent Attention. **vLLM cannot serve this model** — use the transformers-based server instead:
```bash
bash scripts/serving_qwen3_mla.sh           # Launches transformers-based OpenAI-compatible server on port 8001
```
- Checkpoint: `/mnt/shared-storage-gpfs2/sfteval/xtuner_saved_model/internvl3.5/ablate_wuyue2/20260331093205/hf-5615`
- Custom modeling code: `modeling_qwen3_vl_mla.py` (loaded via `trust_remote_code=True`)
- MLA uses `kv_a_proj_with_mqa` + `kv_b_proj` instead of standard `q/k/v_proj` — incompatible with vLLM's built-in Qwen3VL handler

## Conventions
- Each time updates are introduced, remember to update README.md / CLAUDE.md and related setup files, if necessary. Trivial changes can be ignored for these files.
