# Repository Guidelines

## Project Structure & Module Organization
`agent/` is the main package. Keep orchestration, schemas, retries, and parallel utilities in `agent/core/`; put reusable processing units in `agent/extensions/skills/`; compose user-facing modes in `agent/extensions/workflows/`. External framework adapters belong in `agent/integrations/`. `server/` contains the FastAPI app, `templates/` backs the web UI, `tests/` holds the pytest suite, and `scripts/` contains model-serving and cluster helpers. Hermes skill assets live in `.agents/skills/`. Runtime artifacts belong in `cache/`; sample inputs live in `media/`.

## Build, Test, and Development Commands
Install dependencies with:
```bash
pip install -r requirements.txt
```
Optional local install:
```bash
python setup.py install
```
Run the unit and integration tests that CI expects:
```bash
pytest tests/
```
Run the CLI locally:
```bash
python -m agent.main analyze youtube "https://www.youtube.com/watch?v=..." --mode detailed
python -m agent.main chat local media/example.mp4
python -m agent.main hermes install-skill
```
Start the API and web UI:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000
```
Model serving and heavier validation are script-driven:
```bash
bash scripts/serving_qwen3_5.sh
python scripts/test_all.py --video-path media/<file>.mp4 --api-base http://localhost:8000/v1
```

## Coding Style & Naming Conventions
Target Python 3.11+. Use 4-space indentation, snake_case for modules/functions, PascalCase for classes, and concise docstrings where behavior is not obvious. Follow the repo’s layering: workflows assemble skills; skills stay focused and standalone; shared types live in `agent/core/schemas.py`. No formatter or linter config is checked in, so match surrounding imports and line structure.

## Testing Guidelines
Add pytest files as `tests/test_<feature>.py` and prefer fast, self-contained coverage like `tests/test_core_features.py`. Mock network, model, and external process calls where possible. Run `pytest tests/` before opening a PR; if you touch serving, GPU/NPU paths, or end-to-end flows, also note the manual command you ran, such as `bash scripts/run_test_gpu.sh --api-base ...`.

## Commit & Pull Request Guidelines
Recent history uses short, focused subjects such as `npu support` and `update usage of each model`. Keep commit titles brief, lowercase, and centered on one change. PRs should include a summary, affected entrypoints or configs, linked issues when relevant, and sample output or screenshots for API/UI changes. For non-trivial behavior changes, update `README.md`, `CLAUDE.md`, `AGENTS.md` and related setup docs in the same PR.

## Configuration & Assets
Keep environment-specific values in `.env`; on GPU cluster flows, load it before job scripts with `set -a && source .env && set +a`. Model and workflow defaults live in `models.yaml`, `workflows.yaml`, and `config.yaml`. Do not commit generated `cache/` outputs or model weights under `models/`; call out external requirements such as `ffmpeg`, `yt-dlp`, or a vLLM endpoint when they affect setup or verification.
Long-video defaults may also include `parallel_segments` and `parallel_asr` in `workflows.yaml`; if you change those behaviors, keep the helper scripts and setup docs aligned.
