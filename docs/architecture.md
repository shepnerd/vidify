# Architecture

Vidify follows a modular pipeline architecture. Video enters through an I/O layer, passes through configurable skill stages, and produces structured analysis output.

## Directory Layout

```
agent/
  core/
    schemas.py          # Pydantic data models (VideoAsset, FrameSet, Transcript, ...)
    orchestrator.py     # Workflow router — dispatches to the right workflow by mode
  extensions/
    models/
      vllm_openai_client.py   # OpenAI-compatible client for vLLM
      direct_model_loader.py  # Local model loading (no server needed)
    skills/             # 24 self-contained processing units
    workflows/          # 7 high-level pipelines built from skills
    storage/            # Persistence layer
    utils/
      cache.py          # SHA1 hashing, directory helpers, JSON I/O
  config.py             # YAML config loader with built-in defaults
  main.py               # Click CLI entry point
server/
  app.py                # FastAPI REST server (port 9000)
```

## Core Data Models (`agent/core/schemas.py`)

| Model | Purpose |
|-------|---------|
| `VideoSource` | Source descriptor (type: youtube/url/local, uri) |
| `VideoMetadata` | Duration, fps, resolution, audio presence |
| `VideoAsset` | Complete video reference — id, source, local_path, cache_dir |
| `FrameItem` | Single frame — id, timestamp, path, caption |
| `FrameSet` | Collection of frames + sampling strategy |
| `FrameStrategy` | Sampling config — type (fps/scene), params |
| `ASRSegment` | Transcription segment with start/end time |
| `Transcript` | Full transcript with language info |
| `TimelineChapter` | Chapter with title, summary, time range |
| `TimelineEvent` | Event with text and evidence references |
| `HighlightClip` | Video segment selected as a highlight |
| `AnalysisResult` | Top-level output container |

## Data Flow

```
Video Source (YouTube / URL / Local)
        │
        ▼
  load_video() ──► VideoAsset (with cache_dir)
        │
        ├──► probe_video()      → VideoMetadata
        ├──► sample_frames()    → FrameSet (scene or fps strategy)
        ├──► caption_frames()   → FrameSet with captions filled
        ├──► extract_audio()    → audio.wav
        ├──► transcribe()       → Transcript (Whisper ASR)
        ├──► build_timeline()   → chapters + events (LLM)
        │
        │  [detailed mode adds:]
        ├──► extract_text()     → OCR results per frame
        ├──► detect_objects()   → YOLO detections per frame
        ├──► analyze_emotions() → audio + visual emotions
        ├──► translate_asr()    → translated transcript
        │
        ▼
  save_analysis() ──► cache/{vid_hash}/analysis.json
        │
        ├──► wf_index()    → FAISS index (embeddings)
        ├──► wf_ask()      → semantic search + LLM answer
        ├──► wf_highlights()→ clips + reel
        └──► wf_report()   → comprehensive report
```

## Cache Structure

Each video is cached under `cache/videos/{sha1(source_type:uri)}/`:

```
cache/videos/a1b2c3.../
  source.mp4          # Downloaded video
  audio.wav           # Extracted audio
  frames/             # Sampled key frames (f_000001.jpg, ...)
  analysis.json       # Complete analysis output
  index_faiss/        # FAISS index + chunk metadata
  highlights/         # Exported clips + reel
  report.json         # Generated report
```

## Model Interfaces

**vLLM (default)** — uses the OpenAI-compatible API (`/v1/chat/completions`, `/v1/embeddings`). The `make_client()` function in `vllm_openai_client.py` creates a standard OpenAI client pointed at the vLLM server.

**Direct model loading** — loads the model and tokenizer locally for single-node inference without a server. Enabled via `--direct-model` CLI flag.

## Orchestrator

`agent/core/orchestrator.py` is the central dispatcher. It receives a mode string and routes to the corresponding workflow:

| Mode | Workflow | Description |
|------|----------|-------------|
| `brief` | `wf_brief` | Quick analysis — frames + captions + ASR + timeline |
| `detailed` | `wf_detailed` | Full analysis — adds OCR, emotions, objects, translation |
| `index` | `wf_index` | Build FAISS semantic index |
| `ask` | `wf_ask` | Question-answering over indexed content |
| `highlights` | `wf_highlights` | Detect and export highlight clips |
| `report` | `wf_report` | Generate comprehensive analysis report |

## Configuration Precedence

1. **CLI / API parameters** (highest priority)
2. **YAML config files** — `models.yaml`, `workflows.yaml`
3. **Built-in defaults** — `agent/config.py`

See [Configuration Guide](configuration.md) for full details.
