# Configuration

VidCopilot uses YAML-based configuration with sensible defaults. Configuration is optional — the system works out of the box with built-in defaults.

## Configuration Files

### models.yaml

Controls model selection and parameters for each processing skill.

```yaml
mllm:
  heavy:
    model_name: "qwen-vl-7b"
    base_url: "http://localhost:8000/v1"
    max_tokens: 512
    temperature: 0.7
  light:
    model_name: "qwen-vl-1b"
    base_url: "http://localhost:8000/v1"
    max_tokens: 256
    temperature: 0.5

ocr:
  engine: "paddleocr"
  lang: "ch"
  use_angle_cls: true

object_detection:
  model: "yolov8n.pt"
  conf_threshold: 0.5

asr:
  model: "whisper"
  size: "small"
  language: null          # null = auto-detect

emotion_analysis:
  audio_model: "wav2vec2-emotion"
  visual_model: "fer"

translation:
  source_lang: "en"
  target_lang: "zh"
  model: "helsinki-nlp"
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

VidCopilot requires a vLLM server providing OpenAI-compatible endpoints.

### Minimum command

```bash
vllm serve /path/to/model \
  --host 0.0.0.0 --port 8000 \
  --allowed-local-media-path /abs/path/to/cache
```

### Important flags

| Flag | Why |
|------|-----|
| `--allowed-local-media-path` | Required. VidCopilot passes local frame paths as `file://` image URLs. |
| `--generation-config vllm` | Recommended. Prevents unexpected behavior from model repo's `generation_config.json`. |
| `--tensor-parallel-size N` | Set to match your GPU count. |
| `--max-model-len 32768` | Needed for long video contexts. |

### GPU cluster serving

On a GPU cluster using `rl.sh` (rlaunch wrapper):

```bash
# Launch a detached 2-GPU serving job
rl.sh -gpu 2 -d -- vllm serve /path/to/model \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --max-model-len 32768
```

The E2E test script can auto-launch serving — see [Testing Guide](testing.md).

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
- `vidcopilot` — FastAPI server on port 9000
- `vllm` — vLLM model server on port 8000 (GPU-enabled)

### Standalone Docker

```bash
docker build -t vidcopilot .
docker run -p 9000:9000 vidcopilot
```
