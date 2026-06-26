# Live Streaming

Vidify can process webcams and RTMP/HTTP streams in real time. The live workflow
maintains rolling memory over incoming frames and supports Q&A while processing
continues.

## Architecture

The streaming pipeline has three main modules:

```text
Frames
  -> Perception
       frame capture, scene detection, SlowFast visual analysis
  -> Memory
       local per-segment memory, global summary, backup-on-query snapshots
  -> Reasoning
       segment retrieval, context building, LLM Q&A
```

## Modules

| Module | Responsibility |
|--------|----------------|
| Perception | Captures frames at configured FPS, detects scene changes with CLIP similarity, and routes frames through heavy or light analysis models |
| Memory | Stores local segment summaries and embeddings, then maintains a global LLM-generated summary across segments |
| Reasoning | Snapshots memory for consistent retrieval, selects relevant segments, and answers questions with evidence |

## Key Features

| Feature | Description |
|---------|-------------|
| Adaptive segmentation | CLIP-based scene-change detection creates semantic segments instead of fixed windows |
| SlowFast analysis | Heavy model work runs every N frames; light model work handles intermediate frames |
| Two-level memory | Local segment memory supports precise retrieval; global summary supports holistic questions |
| Live Q&A | Questions can be asked while stream processing continues |
| Backup-on-query | Memory is copied before retrieval so a query sees a consistent state |

## CLI Usage

```bash
# Webcam
python -m agent.main analyze local webcam --mode live

# RTMP stream
python -m agent.main analyze local stream --mode live \
  --stream-source stream --stream-url rtmp://host/live/key

# HTTP stream
python -m agent.main analyze local stream --mode live \
  --stream-source stream --stream-url http://camera-ip/video
```

## REST API

Start the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000
```

Start a session:

```bash
curl -X POST http://localhost:9000/live/start \
  -H 'Content-Type: application/json' \
  -d '{"source":"stream","stream_url":"rtmp://host/live/key","fps":1}'
```

Ask a question:

```bash
curl -X POST http://localhost:9000/live/ask \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"live_0001","question":"What is happening in the video?"}'
```

Check status:

```bash
curl http://localhost:9000/live/status/live_0001
```

Stop and return final memory:

```bash
curl -X POST http://localhost:9000/live/stop \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"live_0001"}'
```

## Configuration

Settings live under `live_stream` in `workflows.yaml`:

```yaml
live_stream:
  source: webcam
  fps: 1
  heavy_interval: 5
  similarity_threshold: 0.9
  min_segment_frames: 3
  max_segment_frames: 16
```

Model tiers live in `models.yaml`:

```yaml
mllm:
  heavy:
    model_name: qwen3.5-9b
    base_url: http://localhost:8000/v1
  light:
    model_name: qwen3.5-4b
    base_url: http://localhost:8000/v1
```

See [Configuration](configuration.md) for precedence rules and endpoint setup.
