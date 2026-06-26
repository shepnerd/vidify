# Production Features

Vidify includes operational patterns for long-running video workflows, optional
dependencies, and model-serving failures.

## Retry with Exponential Backoff

Model calls such as vLLM chat, Whisper ASR, and embedding requests are wrapped
with retry handling for transient failures: timeouts, connection errors, 5xx
responses, and rate limits.

```python
from agent.core.retry import retry_with_backoff

@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0)
def my_api_call():
    ...
```

## Graceful Degradation

Optional skills such as OCR, object detection, emotion analysis, translation,
and web search use `@skill_guard`. If an optional dependency is missing or a
model call fails, the skill is skipped and the workflow continues with a warning.

This matters because many environments do not have every optional dependency:
GPU/NPU runtimes, OCR engines, YOLO weights, internet access, or model-serving
endpoints may be unavailable during fast tests.

## Parallel Skill Execution

The `detailed` workflow runs independent skills such as OCR, object detection,
and emotion analysis in a thread pool. Configure concurrency in `workflows.yaml`:

```yaml
detailed:
  max_parallel_skills: 3
```

## Parallel Segment Processing

For long videos, `brief` and `detailed` can split work into temporal segments,
process segments concurrently, and merge timestamps back into a unified result.

```text
Long video
  -> split into temporal segments
  -> process frames, captions, OCR, detection, and emotion per segment
  -> adjust timestamps
  -> merge results
  -> build final timeline
```

Global steps stay global: probing, sufficiency checks, timeline construction,
translation, and web search. Segment-local steps include frame sampling, visual
captioning, OCR, object detection, and emotion analysis.

Example configuration:

```yaml
detailed:
  parallel_segments:
    enabled: true
    segment_duration: 300
    max_workers: 4
    min_video_duration: 300
    min_segment_duration: 30
  parallel_asr:
    enabled: true
    max_workers: 4
    segment_duration: 240
    min_audio_duration: 300
    min_segment_duration: 30
```

## Pluggable Segmentation

Segmentation strategies implement `BaseSegmentor` in `agent/core/segment.py`.
The default `DurationSegmentor` uses fixed-duration FFmpeg ranges. Custom
segmentors can register scene, semantic, or model-based boundaries:

```python
from agent.core.segment import BaseSegmentor, register_segmentor

class SceneSegmentor(BaseSegmentor):
    def segment(self, video_path, duration_sec, base_cache_dir):
        boundaries = my_model.predict(video_path)
        segments = []
        for i, (start, end) in enumerate(boundaries):
            segments.append(self._make_segment(i, start, end, base_cache_dir))
        return self._merge_tiny_tail(segments, duration_sec)

register_segmentor("scene", SceneSegmentor)
```

Set `segmentor_name: scene` in config, or pass it to
`split_video_into_segments()`.

## Progress Events

`agent.core.events` emits lifecycle events:

| Event Type | Purpose |
|------------|---------|
| `skill_start` | A skill or workflow step started |
| `skill_complete` | A skill or workflow step completed |
| `skill_error` | A skill failed |
| `skill_skipped` | A guarded optional skill was skipped |
| `progress` | Human-readable progress update |

The CLI prints progress to stderr. The API exposes progress through
`POST /analyze/stream` as Server-Sent Events.

## Lifecycle Hooks

Shell commands can run at workflow milestones through `hooks.yaml`:

```yaml
hooks:
  post_analysis:
    - command: "curl -X POST $WEBHOOK_URL -d @$RESULT_PATH"
      async: true
      timeout: 10
  on_error:
    - command: "echo 'Failed: $ERROR_MSG' >> errors.log"
```

Hook points include `pre_analysis`, `post_analysis`, `post_skill`, `on_error`,
`post_highlight`, and `post_index`.

## Structured Logging

Pass `--log-format json` to the CLI for machine-readable logs with fields such
as `video_id`, `skill_name`, `duration_ms`, and `status`.

`WorkflowTracker` can be used for per-workflow skill timing summaries.
