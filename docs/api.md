# API Reference

Vidify provides both a REST API (FastAPI server) and a CLI interface.

## REST API

Start the server:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000
```

Interactive docs at `http://localhost:9000/docs` (Swagger UI).

### Endpoints

#### `GET /health`
Health check.
```json
{"ok": true}
```

#### `POST /analyze`
Run video analysis (brief or detailed).

```bash
curl -X POST http://localhost:9000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type": "youtube",
    "uri": "https://www.youtube.com/watch?v=XXXX",
    "mode": "detailed",
    "cache_root": "./cache",
    "llm_base_url": "http://localhost:8000/v1",
    "llm_model": "qwen-vl",
    "max_frames": 128,
    "whisper_model": "small"
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source_type` | `youtube\|url\|local` | required | Video source type |
| `uri` | string | required | YouTube URL, HTTP URL, or local path |
| `mode` | `quick\|detailed` | `detailed` | Analysis depth |
| `cache_root` | string | `./cache` | Cache directory |
| `llm_base_url` | string | `http://localhost:8000/v1` | vLLM endpoint |
| `llm_model` | string | `qwen-vl` | Model name |
| `max_frames` | int (1-128) | 128 | Max frames to sample |
| `whisper_model` | string | `small` | Whisper model size |
| `direct_model` | bool | false | Use local model loading |
| `model_path` | string | `/models/qwen-vl` | Path for direct model |

#### `POST /index`
Build FAISS semantic index (requires prior analysis or auto-generates one).

```bash
curl -X POST http://localhost:9000/index \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type": "youtube",
    "uri": "https://www.youtube.com/watch?v=XXXX",
    "cache_root": "./cache",
    "llm_base_url": "http://localhost:8000/v1",
    "llm_model": "qwen-vl",
    "embed_base_url": "http://localhost:8000/v1",
    "embed_model": "qwen-embed",
    "chunk_sec": 20
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embed_base_url` | string | `http://localhost:8000/v1` | Embedding API endpoint |
| `embed_model` | string | `qwen-embed` | Embedding model name |
| `chunk_sec` | int (5-120) | 20 | Time chunk size in seconds |

#### `POST /ask`
Question-answering over indexed video content.

```bash
curl -X POST http://localhost:9000/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type": "youtube",
    "uri": "https://www.youtube.com/watch?v=XXXX",
    "cache_root": "./cache",
    "question": "What are the key conclusions?",
    "top_k": 5,
    "llm_base_url": "http://localhost:8000/v1",
    "llm_model": "qwen-vl",
    "embed_base_url": "http://localhost:8000/v1",
    "embed_model": "qwen-embed"
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | Natural-language question |
| `top_k` | int (1-20) | 5 | Number of chunks to retrieve |

#### `POST /highlights`
Detect and export highlight clips.

```bash
curl -X POST http://localhost:9000/highlights \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type": "youtube",
    "uri": "https://www.youtube.com/watch?v=XXXX",
    "cache_root": "./cache",
    "llm_base_url": "http://localhost:8000/v1",
    "llm_model": "qwen-vl",
    "max_clips": 5,
    "also_make_reel": true
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_clips` | int (1-20) | 5 | Maximum highlight clips |
| `also_make_reel` | bool | true | Concatenate clips into reel |

#### `POST /analysis`
Retrieve cached analysis results.

```bash
curl -X POST http://localhost:9000/analysis \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type": "youtube",
    "uri": "https://www.youtube.com/watch?v=XXXX",
    "cache_root": "./cache"
  }'
```

#### `GET /`
Web GUI for video upload and analysis.

#### `POST /upload`
Upload a local video file through the web interface.

---

## CLI

```bash
python agent/main.py SOURCE_TYPE URI [OPTIONS]
```

### Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `SOURCE_TYPE` | `youtube`, `url`, `local` | Video source type |
| `URI` | string | YouTube URL, HTTP URL, or file path |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `detailed` | `brief`, `detailed`, `highlights`, `index`, `ask`, `report` |
| `--cache-root` | `./cache` | Cache directory |
| `--question` | — | Question for `ask` mode |
| `--max-frames` | 128 | Maximum frames to sample |
| `--interactive` | false | Interactive prompt mode |
| `--direct-model` | false | Use local model loading |
| `--model-path` | `/models/qwen-vl` | Model path for direct loading |
| `--include-web-search` | false | Enable web search enhancement |
| `--google-api-key` | — | Google Custom Search API key |
| `--google-search-engine-id` | — | Google Custom Search engine ID |

### Examples

```bash
# Detailed analysis of a YouTube video
python agent/main.py youtube "https://www.youtube.com/watch?v=..." --mode detailed

# Brief analysis with web search
python agent/main.py youtube "https://www.youtube.com/watch?v=..." --mode brief --include-web-search

# Question-answering (runs index + ask)
python agent/main.py youtube "https://www.youtube.com/watch?v=..." --mode ask \
    --question "What are the main arguments?"

# Local video with direct model
python agent/main.py local /path/to/video.mp4 --mode detailed --direct-model \
    --model-path /path/to/qwen-vl

# Generate report
python agent/main.py youtube "https://www.youtube.com/watch?v=..." --mode report
```
