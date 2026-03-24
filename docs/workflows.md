# Workflows

Workflows are high-level pipelines that compose skills into end-to-end processing chains. They live in `agent/extensions/workflows/`.

## Overview

| Workflow | File | Input | Output |
|----------|------|-------|--------|
| Brief | `brief.py` | VideoAsset | Metadata, frames, ASR, timeline |
| Detailed | `detailed.py` | VideoAsset | All of brief + OCR, emotions, objects, translation |
| Index | `index.py` | VideoAsset (needs analysis) | FAISS index + chunk metadata |
| Ask | `ask.py` | VideoAsset + question (needs index) | Answer with evidence |
| Highlights | `highlights.py` | VideoAsset (needs analysis) | Clips + optional reel |
| Report | `report.py` | VideoAsset | Structured analysis report |
| Analyze | `analyze.py` | VideoAsset + mode | Routes to brief or detailed |

## Brief Workflow

Fast analysis for quick video insights.

**Steps:**
1. Probe video metadata (duration, resolution, fps)
2. Sample frames using scene detection (threshold: 0.3, max: 64)
3. Caption frames with vision model
4. Extract audio and run Whisper ASR
5. Build structured timeline (chapters + events) via LLM
6. Optionally enhance with web search

**Parameters:**
- `max_frames` — default 64
- `whisper_model` — default "small" (set `None` to skip ASR)
- `include_web_search` — default False
- `direct_model` / `model_path` — for local model loading

## Detailed Workflow

Comprehensive analysis with all available skills.

**Adds to brief:**
- OCR text extraction from key frames (PaddleOCR)
- Object detection (YOLOv8) on frames
- Emotion analysis — audio (Wav2Vec2) + visual (FER)
- ASR translation to target language (default: Chinese)
- Lower scene detection threshold (0.25) and more frames (128)

**Parameters:**
- `max_frames` — default 128
- All brief parameters plus advanced skill toggles

## Index Workflow

Builds a FAISS semantic index for retrieval-augmented Q&A.

**Steps:**
1. Load existing analysis (or auto-run detailed if missing)
2. Chunk video content by time windows (default: 20s)
3. Generate dense embeddings via OpenAI-compatible API
4. Build FAISS index with L2-normalized vectors

**Parameters:**
- `chunk_sec` — default 20
- `embed_base_url` / `embed_model` — embedding API endpoint

**Output:** Index files in `cache/{vid}/index_faiss/`, item count, chunk metadata.

## Ask Workflow

Answers natural-language questions about video content using semantic search.

**Steps:**
1. Embed the question using the same embedding model
2. Search FAISS index for top-k relevant chunks
3. Synthesize an answer using LLM with retrieved context

**Parameters:**
- `question` — the question to answer
- `top_k` — default 5

**Output:**
```json
{
  "result": {
    "answer": "...",
    "evidence": [{"start": 10.0, "end": 30.0, "frame_ids": [...], ...}]
  },
  "hits": [...]
}
```

## Highlights Workflow

Detects high-impact segments and exports video clips.

**Steps:**
1. Load analysis (or auto-run detailed if missing)
2. Detect highlights based on information density (LLM scoring)
3. Export individual clips via FFmpeg
4. Optionally concatenate into a highlight reel

**Parameters:**
- `max_clips` — default 5
- `also_make_reel` — default True

**Output:** Clip paths, reel path, timeline mapping.

## Report Workflow

Generates a comprehensive analysis report combining all sources.

**Steps:**
1. Load or run analysis
2. Extract key information: metadata, timeline, top frames, transcript
3. Perform web search if enabled
4. Generate intelligent recommendations via LLM

**Output:** Structured report JSON with sections for metadata, timeline summary, key frames, transcript highlights, web search insights, and recommendations.

## Dependency Chain

```
brief / detailed   (standalone — no prerequisites)
       │
       ▼
     index         (requires analysis.json)
       │
       ▼
      ask          (requires FAISS index)

highlights         (requires analysis.json)
report             (requires analysis.json or runs brief internally)
```

Missing prerequisites are auto-generated when possible (e.g., `index` will run `detailed` if no `analysis.json` exists).
