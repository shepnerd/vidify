---
name: vidify
description: Analyze, search, and summarize videos from YouTube URLs, HTTP URLs, or local files using Vidify's CLI and API.
homepage: https://github.com/shepnerd/vidcopilot
metadata:
  {
    "hermes": {
      "category": "media",
      "requires": { "bins": ["ffmpeg"] },
      "preferredEntryPoints": ["scripts/vidify-analyze.sh", "scripts/vidify-ask.sh"]
    }
  }
---

# Vidify

Use Vidify when the task is about understanding video content end-to-end:

- detailed video summaries
- transcript extraction
- timestamped Q&A
- searchable indexing
- highlight detection
- report generation

## Inputs

- `source_type`: `youtube`, `url`, or `local`
- `uri`: YouTube URL, HTTP URL, or local file path
- `mode`: `brief`, `detailed`, `index`, `ask`, `highlights`, `report`
- `question`: required for `ask`

## Primary commands

Analyze a video:

```bash
bash {baseDir}/scripts/vidify-analyze.sh <source_type> "<uri>" [mode]
```

Ask a question about a video:

```bash
bash {baseDir}/scripts/vidify-ask.sh <source_type> "<uri>" "<question>"
```

Run the API server when repeated calls or concurrent access are needed:

```bash
bash {baseDir}/scripts/vidify-server.sh start
```

## Execution notes

- Prefer `brief` for first-pass understanding.
- Use `detailed` when OCR, emotion analysis, or object detection matter.
- Use `ask` for evidence-backed questions after or instead of a summary.
- Vidify caches work under `./cache`, so repeated calls on the same video are cheaper.
- These wrappers prefer the installed `vidify` CLI, but fall back to `python -m agent.main` from the repo checkout.

## Requirements

- Python 3.11+
- `ffmpeg` on `PATH`
- a multimodal OpenAI-compatible endpoint, by default `http://localhost:8000/v1`

## Examples

```bash
bash {baseDir}/scripts/vidify-analyze.sh youtube "https://www.youtube.com/watch?v=VIDEO_ID" brief
bash {baseDir}/scripts/vidify-analyze.sh local "/path/to/video.mp4" detailed
bash {baseDir}/scripts/vidify-ask.sh youtube "https://www.youtube.com/watch?v=VIDEO_ID" "What are the main conclusions?"
```
