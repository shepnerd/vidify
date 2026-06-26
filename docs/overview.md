# Project Overview

Vidify turns video sources into structured, searchable, and actionable analysis.
It accepts YouTube URLs, HTTP video URLs, local files, webcams, and live streams.

## Capability Map

| Area | What Vidify Does |
|------|------------------|
| Analysis | Probes media, extracts subtitles and metadata, runs ASR fallback, and builds timelines |
| Visual understanding | Samples frames, captions visual content, extracts OCR text, detects objects, and analyzes emotion |
| Retrieval | Builds FAISS indexes over frames, transcript, metadata, and timeline chunks |
| Q&A | Answers natural-language questions with retrieved evidence and targeted visual lookup |
| Editing | Detects highlight segments and exports clips or reels |
| Reporting | Generates structured reports, optionally enriched with web search context |
| Streaming | Maintains live memory over webcams or RTMP/HTTP streams and supports mid-stream Q&A |

## ASR-First Design

Most videos communicate their main information through speech, subtitles, titles,
and descriptions. Vidify uses visual model calls only when they add value.

1. Subtitles first - for YouTube and web videos, embedded manual or auto-generated subtitles are extracted with `yt-dlp`.
2. ASR fallback - if subtitles are unavailable, Whisper transcribes audio.
3. Metadata context - title, description, tags, and uploader information are included in downstream prompts.
4. Sufficiency check - a fast heuristic checks speech coverage and word count before expensive visual captioning.
5. Conditional visual processing - frame captioning runs for silent videos, music videos, sparse-speech media, or forced visual analysis.
6. Targeted visual lookup - visual Q&A captions only frames near relevant timestamps when possible.

The default sufficiency thresholds are configured through `workflows.yaml`:

```yaml
brief:
  asr_first: true
  min_coverage_ratio: 0.3
  min_word_count: 50
  force_visual: false
  prefer_subtitles_over_asr: true
```

## Processing Flow

```text
Source video
  -> download or load local media
  -> probe duration, fps, resolution, and metadata
  -> extract subtitles if present
  -> run ASR if subtitles are missing or insufficient
  -> check transcript sufficiency
  -> skip or run visual captioning
  -> build timeline from transcript, metadata, and selected frames
  -> persist analysis in cache
```

`ask`, `highlights`, and `report` reuse cached analysis when possible. `index`
builds the retrieval layer used by Q&A.

## Project Structure

| Path | Role |
|------|------|
| `agent/core/` | Shared contracts, orchestration, events, hooks, retries, segmenting, and parallel execution |
| `agent/extensions/skills/` | Standalone video/audio/retrieval/analysis units |
| `agent/extensions/workflows/` | End-to-end modes composed from skills |
| `agent/extensions/models/` | vLLM/OpenAI-compatible clients and direct model loading |
| `agent/integrations/` | External framework adapters, including Hermes |
| `agent/extensions/mra/` | Meta-Reflective Auditor implementation |
| `server/` | FastAPI app, REST endpoints, SSE progress streaming, and web UI routes |
| `scripts/` | Serving, validation, demos, and local workflow helpers |
| `docs/` | Architecture, API, workflow, deployment, and testing docs |
| `cache/` | Runtime outputs such as downloads, frames, audio, indexes, and analysis JSON |

Runtime artifacts, model weights, extracted frames/audio, logs, FAISS indexes,
and uploaded files should stay out of Git.
