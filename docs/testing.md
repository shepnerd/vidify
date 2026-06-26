# Testing Guide

Use fast pytest coverage for normal development and script-based validation when
a model endpoint, hardware runtime, external downloader, API route, or web UI
path is part of the touched code.

## Fast Test Suite

```bash
pytest tests/
```

Focused tests while iterating:

```bash
pytest tests/test_core_features.py
pytest tests/test_parallel_segments.py
pytest tests/test_parallel_asr.py
pytest tests/test_mra.py
pytest tests/test_web_search.py
```

Prefer mocks for network, model-serving, FFmpeg-heavy work, external processes,
and hardware-specific paths.

## Local Video Validation

`scripts/test_all.py` runs the 17-skill validation suite against a local video.
The most predictable path is to pass an existing OpenAI-compatible endpoint.

```bash
# Use an existing endpoint
python scripts/test_all.py \
  --video-path media/taste_in_china_s1e1.mp4 \
  --api-base http://localhost:8000/v1

# Specific tests only
python scripts/test_all.py \
  --video-path media/taste_in_china_s1e1.mp4 \
  --api-base http://localhost:8000/v1 \
  --tests frames qa highlights
```

Tests: `video_probe` | `frame_sample` | `audio_extract` | `asr` | `ocr` |
`object_detection` | `subtitle_parse` | `metadata_extract` |
`content_sufficiency` | `needs_visual` | `asr_first_brief` | `frame_caption` |
`video_caption` | `timeline` | `video_qa` | `highlights` | `video_edit`

## GPU Endpoint Validation

`scripts/run_test_gpu.sh` validates against an already-running GPU-backed
OpenAI-compatible endpoint. It does not start a managed GPU job.

```bash
bash scripts/run_test_gpu.sh \
  --api-base http://localhost:8000/v1 \
  --video media/my_video.mp4

bash scripts/run_test_gpu.sh \
  --api-base http://localhost:8000/v1 \
  --video media/my_video.mp4 \
  --tests "frame_caption video_qa highlights"
```

Start vLLM separately with your local process manager or GPU scheduler. See
[Deployment](deployment.md).

## Ascend / NPU Endpoint Validation

`scripts/run_test_ascend.sh` has the same contract for Ascend/NPU-backed vLLM
services:

```bash
bash scripts/run_test_ascend.sh \
  --api-base http://localhost:8000/v1 \
  --video media/my_video.mp4
```

Provider-specific scheduler commands, internal registries, and mount paths should
stay in local docs or `.env` files.

## YouTube E2E

`scripts/test_youtube_e2e.py` downloads a YouTube video and runs a smaller suite
of end-to-end checks. It requires internet access on the current node.

```bash
# Auto-detect or launch serving, then run all tests
python scripts/test_youtube_e2e.py

# Use an existing endpoint
python scripts/test_youtube_e2e.py --api-base http://localhost:8000/v1

# Custom video and selected tests
python scripts/test_youtube_e2e.py \
  --youtube "https://www.youtube.com/watch?v=..." \
  --tests frames qa multi_turn_qa
```

Tests: `frames` | `batch_frames` | `video_caption` | `qa` | `multi_turn_qa`

Results are saved under `cache/`.

## API and Web UI Validation

When changing `server/`, request schemas, SSE progress, uploads, or templates,
run the app and exercise the changed path:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000
```

Example request:

```bash
curl -X POST http://localhost:9000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"local","uri":"media/example.mp4","mode":"brief"}'
```

## Other Script Helpers

| Script | Purpose |
|--------|---------|
| `scripts/demo.py` | End-to-end demo: analyze, index, ask, highlights |
| `scripts/local_video_summary.py` | Local video analysis without the API server |
| `scripts/test_brief.py` | Brief workflow API validation |
| `scripts/test_detailed.py` | Detailed workflow API validation |
| `scripts/test_index.py` | FAISS index API validation |
| `scripts/test_ask.py` | Q&A API validation |
| `scripts/test_highlights.py` | Highlight export API validation |
| `scripts/demo_multi_region_search.py` | Multi-region web-search demo |

## Manual Validation Notes

Document skipped heavy validation in PRs with the command someone should run
later. This is especially important for serving scripts, GPU/NPU startup, vLLM
model behavior, YouTube downloads, live streams, and full workflow changes.
