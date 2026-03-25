# VidCopilot REST API Reference

## Base URL

`http://localhost:9000` (configurable via `VIDCOPILOT_PORT` env var)

## Endpoints

### GET /health

Health check.

**Response:** `{"ok": true}`

### POST /analyze

Run video analysis (quick or detailed).

**Request body:**

```json
{
  "source_type": "youtube",
  "uri": "https://www.youtube.com/watch?v=VIDEO_ID",
  "mode": "detailed",
  "cache_root": "./cache",
  "llm_base_url": "http://localhost:8000/v1",
  "llm_model": "qwen-vl",
  "max_frames": 128,
  "whisper_model": "small"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source_type` | `youtube\|url\|local` | required | Video source type |
| `uri` | string | required | Video URL or file path |
| `mode` | `quick\|detailed` | `detailed` | Analysis depth |
| `cache_root` | string | `./cache` | Cache directory |
| `llm_base_url` | string | `http://localhost:8000/v1` | vLLM endpoint |
| `llm_model` | string | `qwen-vl` | Vision-language model name |
| `max_frames` | int (1-128) | `128` | Max frames to sample |
| `whisper_model` | string | `small` | Whisper ASR model size |

### POST /index

Build FAISS semantic search index from analysis. Auto-runs detailed analysis if not cached.

**Request body:**

```json
{
  "source_type": "youtube",
  "uri": "https://www.youtube.com/watch?v=VIDEO_ID",
  "cache_root": "./cache",
  "embed_base_url": "http://localhost:8000/v1",
  "embed_model": "qwen-embed",
  "chunk_sec": 20
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embed_base_url` | string | `http://localhost:8000/v1` | Embeddings endpoint |
| `embed_model` | string | `qwen-embed` | Embedding model name |
| `chunk_sec` | int (5-120) | `20` | Time window per chunk (seconds) |

### POST /ask

Ask a question about a video. Requires index (auto-builds if missing).

**Request body:**

```json
{
  "source_type": "youtube",
  "uri": "https://www.youtube.com/watch?v=VIDEO_ID",
  "question": "What does the speaker say about performance?",
  "top_k": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | The question to answer |
| `top_k` | int (1-20) | `5` | Number of retrieval hits |

### POST /highlights

Detect and export highlight clips. Auto-runs detailed analysis if not cached.

**Request body:**

```json
{
  "source_type": "youtube",
  "uri": "https://www.youtube.com/watch?v=VIDEO_ID",
  "max_clips": 5,
  "also_make_reel": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_clips` | int (1-20) | `5` | Max highlight clips |
| `also_make_reel` | bool | `true` | Concatenate clips into a reel |

### POST /analysis

Load cached analysis results (no processing).

**Request body:**

```json
{
  "source_type": "youtube",
  "uri": "https://www.youtube.com/watch?v=VIDEO_ID",
  "cache_root": "./cache"
}
```

Returns 404 if no cached analysis exists for the given video.
