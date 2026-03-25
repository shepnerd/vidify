# VidCopilot OpenClaw Skill

An [OpenClaw](https://openclaw.ai) skill that integrates [VidCopilot](https://github.com/user/vidcopilot) — a video understanding agent for deep video analysis, transcription, Q&A, highlight detection, and more.

## Installation

### 1. Install VidCopilot and dependencies

```bash
# From PyPI (when published)
pip install vidcopilot

# Or from source
git clone https://github.com/user/vidcopilot.git
cd vidcopilot
pip install -e .
```

System dependencies:
- **Python 3.11+**
- **ffmpeg** — `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)
- **vLLM** — a running vLLM server with a multimodal model (e.g., Qwen3-VL) for captioning and Q&A

### 2. Install the skill into OpenClaw

Copy this directory to your OpenClaw workspace:

```bash
cp -r /path/to/vidcopilot/openclaw ~/.openclaw/workspace/skills/vidcopilot
```

Or symlink it:

```bash
ln -s /path/to/vidcopilot/openclaw ~/.openclaw/workspace/skills/vidcopilot
```

### 3. Restart OpenClaw

```bash
/new
# or
openclaw gateway restart
```

### 4. Verify

```bash
openclaw skills list
# Should show "vidcopilot" in the list
```

## Usage

Once installed, simply ask the OpenClaw agent to analyze videos:

- "Analyze this YouTube video: https://www.youtube.com/watch?v=..."
- "What are the highlights of this video?"
- "Transcribe this video and summarize it"
- "What does the speaker say about topic X?"

## Configuration

The skill uses these defaults which can be adjusted:

| Setting | Default | Description |
|---------|---------|-------------|
| vLLM endpoint | `http://localhost:8000/v1` | Set via `--config` or env vars |
| Cache directory | `./cache` | Where video assets are cached |
| Max frames | `128` | Maximum frames to sample per video |
| Whisper model | `small` | ASR model size (tiny/small/medium/large) |

## Optional: REST API mode

For concurrent access or integration with other tools, start the VidCopilot API server:

```bash
~/.openclaw/workspace/skills/vidcopilot/scripts/vidcopilot-server.sh start
```

The server runs on port 9000 (configurable via `VIDCOPILOT_PORT`).
