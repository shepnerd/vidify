# Configuration

Vidify uses YAML-based configuration with sensible defaults. Configuration is optional — the system works out of the box with built-in defaults.

## Configuration Files

### models.yaml

Controls model selection and parameters for each processing skill.

```yaml
mllm:
  heavy:
    model_name: "qwen3.5-9b"
    base_url: "http://localhost:8000/v1"
    max_tokens: 512
    temperature: 0.7
  light:
    model_name: "qwen3.5-4b"
    base_url: "http://localhost:8000/v1"
    max_tokens: 256
    temperature: 0.5

ocr:
  engine: "paddleocr"
  lang: "ch"
  use_angle_cls: true

object_detection:
  model: "models/yolov8n.pt"
  conf_threshold: 0.5

asr:
  model: "whisper"
  size: "small"
  language: null          # null = auto-detect

emotion_analysis:
  audio_model: "models/wav2vec2-base-superb-er"
  visual_model: "fer"

translation:
  source_lang: "en"
  target_lang: "zh"
  model: "models/opus-mt-en-zh"
```

### workflows.yaml

Controls workflow behavior and step parameters.

```yaml
brief:
  use_asr: true
  max_frames: 64
  include_web_search: false

detailed:
  use_advanced_skills: true
  max_frames: 128
  heavy_interval: 5

live_stream:
  source: "webcam"
  resolution: [640, 480]
  fps: 1
  heavy_interval: 5
```

## Precedence

Parameters resolve in this order (highest to lowest):

1. **CLI / API arguments** — always override everything
2. **YAML config files** — `models.yaml` and `workflows.yaml`
3. **Built-in defaults** — hardcoded in `agent/config.py`

If no YAML file exists, built-in defaults are used.

## vLLM Setup

Vidify requires a vLLM server providing OpenAI-compatible endpoints.

### Minimum command

```bash
vllm serve /path/to/model \
  --host 0.0.0.0 --port 8000 \
  --allowed-local-media-path /abs/path/to/cache
```

### Important flags

| Flag | Why |
|------|-----|
| `--allowed-local-media-path` | Required. Vidify passes local frame paths as `file://` image URLs. |
| `--generation-config vllm` | Recommended. Prevents unexpected behavior from model repo's `generation_config.json`. |
| `--tensor-parallel-size N` | Set to match your GPU count. |
| `--max-model-len 32768` | Needed for long video contexts. |

### Managed GPU/NPU serving

For managed GPU or NPU environments, start vLLM with your platform's scheduler
and pass the OpenAI-compatible endpoint to Vidify:

```bash
python -m agent.main analyze local video.mp4 \
  --mode detailed \
  --config config.yaml
```

Set `LLM_BASE_URL` and `LLM_MODEL` in `.env` or `config.yaml`, or pass
`--api-base` to the validation scripts.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` | Google Custom Search API key |
| `GOOGLE_SEARCH_ENGINE_ID` | Google Custom Search engine ID |

See [GOOGLE_SEARCH_SETUP.md](../GOOGLE_SEARCH_SETUP.md) for web search setup.

## Docker Deployment

### Docker Compose

```bash
docker-compose up
```

This starts:
- `vidify` — FastAPI server on port 9000
- `vllm` — vLLM model server on port 8000 (GPU-enabled)

### Standalone Docker

```bash
docker build -t vidify .
docker run -p 9000:9000 vidify
```
