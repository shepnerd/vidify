# Testing Guide

## End-to-End YouTube Test (`scripts/test_youtube_e2e.py`)

The primary test script for validating the full video understanding pipeline. It handles model serving discovery/launch, YouTube video download, and runs a suite of tests against the model.

### What it does

1. **Find or launch model serving** — probes known endpoints; if none found, launches a GPU job via `rl.sh` and waits for it to be ready
2. **Download YouTube video** — uses `yt-dlp` (requires internet on the current node)
3. **Run 5 video understanding tests** — exercises different model interaction patterns

### Test Suite

| Test | Description |
|------|-------------|
| `frames` | Extract frames via FFmpeg, caption each individually via image input |
| `batch_frames` | Send multiple frames in a single request, expect structured JSON response |
| `video_caption` | Native video input to Qwen3-VL, segment-by-segment captioning |
| `qa` | Ask a question with video as context |
| `multi_turn_qa` | Two-turn conversation: summary followed by a follow-up question |

### Usage

```bash
# Full auto: detect/launch serving + run all tests
python scripts/test_youtube_e2e.py

# Use an existing serving endpoint
python scripts/test_youtube_e2e.py --api-base http://10.0.0.5:8000/v1

# Custom YouTube video and question
python scripts/test_youtube_e2e.py \
    --youtube "https://www.youtube.com/watch?v=VIDEO_ID" \
    --question "What is being discussed in this video?"

# Launch serving with 4 GPUs
python scripts/test_youtube_e2e.py --gpu 4

# Run only specific tests
python scripts/test_youtube_e2e.py --tests frames qa

# Skip auto-launch (fail if no serving found)
python scripts/test_youtube_e2e.py --skip-serve
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--api-base` | auto-detect | vLLM base URL (e.g. `http://10.0.0.5:8000/v1`) |
| `--youtube` | sample video | YouTube URL to test with |
| `--question` | general summary | Question for the Q&A test |
| `--gpu` | 2 | Number of GPUs for serving |
| `--tp` | same as `--gpu` | Tensor parallel size |
| `--tests` | all 5 | Which tests to run |
| `--skip-serve` | false | Don't auto-launch; fail if no service found |

### How auto-serving works

When no `--api-base` is specified, the script:

1. Checks for a previously launched serving IP in `cache/.serving/serving_ip.txt`
2. Probes `http://{previous_ip}:8000/v1`
3. Probes `http://localhost:8000/v1`
4. If nothing responds, launches a new GPU job:
   - Runs `rl.sh -gpu N -d` to start a detached GPU worker
   - The GPU worker writes its IP to the shared filesystem
   - The GPU worker starts `vllm serve` with the Qwen3-VL model
5. Polls until the vLLM `/v1/models` endpoint responds (up to 10 minutes)

This design works because the shared GPFS filesystem is accessible from both the internet-connected launcher node and the GPU worker node.

### Output

Results are saved to `cache/test_youtube_e2e_results.json`:

```json
{
  "base_url": "http://10.0.0.5:8000/v1",
  "model": "Qwen3-VL-8B-Instruct",
  "youtube": "https://www.youtube.com/watch?v=...",
  "video_path": "cache/videos/abc123/source.mp4",
  "duration": 120.5,
  "passed": 5,
  "failed": 0
}
```

---

## Other Test Scripts

All in `scripts/`. These use the FastAPI server (must be running on port 9000).

### demo.py — Full Pipeline Demo

End-to-end demo: analyze → index → ask → highlights.

```bash
python scripts/demo.py --youtube "https://www.youtube.com/watch?v=..." \
    --llm-base-url http://localhost:8000/v1 --llm-model qwen-vl
```

### local_video_summary.py — Local Video Analysis

Analyze a local video file without a server.

```bash
# Detailed analysis with direct model loading
python scripts/local_video_summary.py \
    --video-path /path/to/video.mp4 \
    --mode detailed \
    --direct-model --model-path /path/to/model

# Brief analysis using vLLM
python scripts/local_video_summary.py \
    --video-path /path/to/video.mp4 \
    --mode brief --max-frames 16
```

### Individual Workflow Tests

These send requests to the FastAPI server (`http://localhost:9000`):

| Script | Workflow | Notes |
|--------|----------|-------|
| `test_brief.py` | Brief analysis | Standalone |
| `test_detailed.py` | Detailed analysis | Standalone |
| `test_index.py` | FAISS indexing | Requires prior analysis |
| `test_ask.py` | Q&A | Requires prior index |
| `test_highlights.py` | Highlight export | Requires prior analysis |
| `test_all.py` | All workflows sequentially | Full pipeline |

### Serving Script

```bash
# Start vLLM serving on 2 GPUs (cluster)
bash scripts/serving_qwen3vl.sh
```

### Web Search Tests

```bash
python tests/test_web_search.py          # Test search functionality
python scripts/demo_multi_region_search.py  # Multi-region demo
```
