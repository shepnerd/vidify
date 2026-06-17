# Repository Guidelines

## Project Structure & Module Organization
`agent/` is the main Python package. Keep orchestration, schemas, events, hooks,
retries, segmenting, and parallel execution in `agent/core/`; put reusable video,
audio, retrieval, and analysis units in `agent/extensions/skills/`; compose
user-facing modes in `agent/extensions/workflows/`. Model adapters belong in
`agent/extensions/models/`, external framework adapters in `agent/integrations/`,
and the Meta-Reflective Auditor in `agent/extensions/mra/`. `server/` contains
the FastAPI app and SSE/live endpoints, `templates/` backs the web UI, `tests/`
holds pytest coverage, `scripts/` contains serving and validation helpers,
`docs/` and `design/` hold architecture notes, and `.agents/skills/` contains the
Hermes skill assets. Runtime artifacts belong in `cache/`; local model weights
belong in `models/`; sample media belongs in `media/`.

Respect the layering when adding behavior: workflows assemble skills, skills stay
focused and standalone, shared data contracts live in `agent/core/schemas.py`,
and config defaults live in `agent/config.py` with overrides from `models.yaml`,
`workflows.yaml`, and optional `config.yaml`. If you add a new workflow mode,
update the CLI choices in `agent/main.py`, routing in `agent/core/orchestrator.py`,
API/request schemas where applicable, docs, and tests together.

## Build, Test, and Development Commands
Install dependencies with:
```bash
pip install -r requirements.txt
```
Optional local install, including the `vidify` console entrypoint:
```bash
python setup.py install
```
Optional extras:
```bash
pip install -e ".[detection]"  # YOLO object detection
pip install -e ".[mra]"        # MRA perception extras
```
Run the fast test suite that CI expects:
```bash
pytest tests/
```
Run focused tests while iterating:
```bash
pytest tests/test_core_features.py
pytest tests/test_parallel_segments.py
pytest tests/test_parallel_asr.py
pytest tests/test_mra.py
pytest tests/test_web_search.py
```
Run the CLI locally:
```bash
python -m agent.main analyze youtube "https://www.youtube.com/watch?v=..." --mode brief
python -m agent.main analyze local media/example.mp4 --mode detailed
python -m agent.main analyze local media/example.mp4 --mode ask --question "What changed?"
python -m agent.main analyze local media/example.mp4 --mode audit
python -m agent.main chat local media/example.mp4
python -m agent.main hermes install-skill
```
Start the API and web UI:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000
```
Start model serving or heavier validation when the touched path needs it:
```bash
bash scripts/serving_qwen3_5.sh
bash scripts/run_test_gpu.sh --api-base http://localhost:8000/v1
bash scripts/run_test_ascend.sh --api-base http://localhost:8000/v1
python scripts/test_all.py --video-path media/<file>.mp4 --api-base http://localhost:8000/v1
python scripts/test_youtube_e2e.py --api-base http://localhost:8000/v1
```
Use Docker only when validating deployment wiring:
```bash
docker-compose up
```

## Coding Style & Naming Conventions
Target Python 3.11+. Use 4-space indentation, snake_case for modules/functions,
PascalCase for classes, and concise docstrings where behavior is not obvious.
Match surrounding imports and line structure; no formatter or linter config is
checked in. Prefer Pydantic models and typed dictionaries already present in the
repo over ad hoc structures. Keep optional integrations guarded with graceful
fallbacks (`skill_guard`, import fallbacks, or clear error messages), because
many tests run without vLLM, GPU/NPU devices, internet, OCR engines, or YOLO
weights.

For video pipeline changes, preserve the ASR-first design: subtitles are preferred,
Whisper ASR is the fallback, the sufficiency check decides whether visual
captioning is needed, and detailed mode still samples frames for OCR/detection.
For parallel long-video work, keep `parallel_segments` and `parallel_asr`
configuration, environment overrides, merge behavior, and timestamp adjustments
aligned across `workflows.yaml`, `agent/core/segment.py`,
`agent/core/segment_worker.py`, `brief.py`, and `detailed.py`.

## Testing Guidelines
Add pytest files as `tests/test_<feature>.py`. Prefer fast, self-contained tests
with mocks for network, model-serving, external processes, FFmpeg-heavy work, and
hardware-specific paths. Run `pytest tests/` before opening a PR. If a change
touches serving scripts, GPU/NPU startup, vLLM model behavior, YouTube download,
live streams, or end-to-end workflows, also run or document the relevant manual
command from `scripts/`.

Test scope should match risk:
- core utilities, schemas, hooks, events, retries: focused unit tests
- workflow composition, ASR-first logic, parallel segments/ASR, MRA: targeted
  tests plus existing regression files
- FastAPI/API or web UI changes: exercise `uvicorn server.app:app --host 0.0.0.0 --port 9000`
  and include sample request/output when behavior changes
- model-serving or cluster scripts: validate with the matching GPU or Ascend
  helper and record the endpoint used

## Configuration, Models, and Assets
Keep environment-specific values in `.env`; start cluster jobs only after loading
it with:
```bash
set -a && source .env && set +a
```
Model and workflow defaults live in `models.yaml`, `workflows.yaml`, and
`config.yaml` when present, with built-in defaults in `agent/config.py`. The
default OpenAI-compatible endpoint is `http://localhost:8000/v1`; vLLM must allow
local media from the cache path, for example `--allowed-local-media-path $(pwd)/cache`.
Qwen3.5 serving uses `scripts/serving_qwen3_5.sh`; legacy Qwen3-VL and Ascend
paths have dedicated scripts.

Do not commit generated runtime data: `cache/`, `agent/extensions/storage/cache/`,
FAISS indexes, extracted frames/audio, uploaded files, logs, model weights under
`models/`, or downloaded media outputs. Keep sample inputs small and intentional.
Call out external requirements such as `ffmpeg`, `ffprobe`, `yt-dlp`, PaddleOCR,
Tesseract, a vLLM endpoint, CUDA/NPU runtime, or Google Custom Search credentials
when they affect setup or verification.

## Documentation Guidelines
For non-trivial behavior changes, update the affected setup and architecture docs
in the same change. Common files are `README.md`, `CLAUDE.md`, `docs/api.md`,
`docs/architecture.md`, `docs/workflows.md`, `docs/skills.md`,
`docs/configuration.md`, and design notes under `design/`. Keep Hermes-facing
behavior aligned with `.agents/skills/media/vidify/SKILL.md` and wrappers under
`.agents/skills/media/vidify/scripts/`.

Document public behavior, config keys, commands, and expected outputs. Avoid
copying large code blocks into docs when a file path or command is more stable.
If docs and code disagree, treat code as source of truth and fix the drift.

## Commit & Pull Request Guidelines
Recent history uses short, focused, lowercase subjects such as `npu support` and
`update usage of each model`. Keep commit titles brief and centered on one change.
PRs should include a summary, affected entrypoints/configs, linked issues when
relevant, tests run, and sample output or screenshots for API/UI changes. Mention
any skipped heavy validation with the reason and the command someone should run
later.
