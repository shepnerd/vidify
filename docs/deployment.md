# Deployment

Vidify runs against OpenAI-compatible model endpoints. The default local setup
uses vLLM on port `8000`, while the FastAPI app runs on port `9000`.

## Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.11+ |
| System tools | `ffmpeg`, `ffprobe`, `yt-dlp` |
| Model serving | vLLM-compatible GPU/NPU endpoint, or direct model loading |
| Default model | Qwen3.5, configurable in `models.yaml` |
| vLLM | `>=0.19.0` for Qwen3.5 |

Optional features may require PaddleOCR, Tesseract, YOLO weights, CUDA/NPU
runtimes, or Google Custom Search credentials.

## Local vLLM Serving

Recommended Qwen3.5 helper:

```bash
pip install "vllm>=0.19.0"
bash scripts/serving_qwen3_5.sh
```

Manual command:

```bash
vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --allowed-local-media-path $(pwd)/cache
```

On multi-GPU hosts:

```bash
TP_SIZE=2 MAX_MODEL_LEN=131072 bash scripts/serving_qwen3_5.sh
```

Legacy Qwen3-VL helper:

```bash
bash scripts/serving_qwen3vl.sh
```

## App Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000
```

The web UI is available at `http://localhost:9000`. Swagger docs are available
at `http://localhost:9000/docs`.

## GPU Endpoint Validation

Use `run_test_gpu.sh` when a GPU-backed OpenAI-compatible endpoint is already
running:

```bash
bash scripts/run_test_gpu.sh --api-base http://localhost:8000/v1 --video media/my_video.mp4

bash scripts/run_test_gpu.sh --api-base http://localhost:8000/v1 \
  --video media/my_video.mp4 --tests "frame_caption video_qa highlights"
```

## Ascend / NPU Serving

Vidify can run against Ascend-backed vLLM deployments through the same
OpenAI-compatible API used for GPU serving. Keep provider-specific scheduler
commands, internal registry URLs, mount paths, and credentials in local docs or
`.env` files instead of committed public docs.

Generic helpers:

```bash
# Qwen3.5-9B
TP_SIZE=2 bash scripts/serving_qwen3_5_ascend.sh /models/Qwen3.5-9B

# Qwen2.5-VL fallback
TP_SIZE=2 bash scripts/serving_qwen2_5vl_ascend.sh /models/Qwen2.5-VL-7B-Instruct

# Validate against an existing Ascend/NPU endpoint
bash scripts/run_test_ascend.sh --api-base http://localhost:8000/v1 --video media/my_video.mp4
```

For Qwen3.5 on Ascend, the helper uses `--enforce-eager` and conservative
`MAX_MODEL_LEN=16384` defaults. Tune `TP_SIZE`, `MAX_MODEL_LEN`, `PORT`, and
`ALLOWED_LOCAL_MEDIA_PATH` for your hardware and vLLM build.

## Docker

Full stack:

```bash
docker-compose up
```

App only:

```bash
docker build -t vidify .
docker run -p 9000:9000 vidify
```

## Local Runtime Environment

Copy `.env.example` when you need overrides:

```bash
cp .env.example .env
```

Common values:

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_BASE_URL` | OpenAI-compatible chat/completions endpoint | `http://localhost:8000/v1` |
| `LLM_MODEL` | Default multimodal model name | `qwen3.5-9b` |
| `EMBED_BASE_URL` | OpenAI-compatible embeddings endpoint | `http://localhost:8000/v1` |
| `EMBED_MODEL` | Default embedding model name | `qwen-embed` |
| `CACHE_ROOT` | Runtime cache directory | `./cache` |

See [Configuration](configuration.md) for YAML config, precedence, and web search
environment variables.
