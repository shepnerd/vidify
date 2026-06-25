# Vidify

[English](README.md)

Vidify 是一个视频理解智能体：输入 YouTube 链接或本地视频，即可生成结构化分析、可检索索引、问答结果、精彩片段和报告。

## 功能概览

| 能力 | 说明 |
|------|------|
| **分析** | 下载视频，提取字幕和元数据，执行 ASR，按需进行帧字幕生成，并构建时间线 |
| **理解** | OCR、目标检测、情绪分析、翻译 |
| **检索** | 基于帧、ASR 和元数据构建 FAISS 索引，支持语义问答和定向视觉查找 |
| **剪辑** | 自动检测精彩片段，导出视频片段，组装短视频 |
| **增强** | Web 搜索上下文、多语言支持 |
| **报告** | 生成完整的视频分析报告 |
| **流式处理** | 支持实时直播流和摄像头处理，带自适应分段和两级记忆 |
| **并行分段** | 将长视频切分为时间片段，并行处理后合并结果 |
| **可靠性** | 指数退避重试、可选能力优雅降级、生命周期钩子 |

## 设计理念：ASR 优先，视觉作为最后手段

大多数视频（纪录片、vlog、演示、访谈、影评、体育解说等）的关键信息主要来自语音或字幕。Vidify 围绕这个事实设计：

1. **字幕优先**：对 YouTube 和 Web 视频，优先通过 yt-dlp 提取内嵌字幕（人工字幕或自动字幕），并作为主要转录文本。字幕通常免费且质量高于 ASR。
2. **ASR 兜底**：没有字幕时，使用 Whisper ASR 转录音频。
3. **元数据上下文**：提取视频标题、描述、标签、上传者等信息，并用于时间线构建和问答。
4. **充分性检查**：使用快速启发式规则（不调用 LLM）判断转录文本是否足够，从而跳过昂贵的 MLLM 视觉处理。默认条件为语音覆盖率不低于 30%，且词数不低于 50。
5. **条件式视觉处理**：只有在转录文本不足时才运行 MLLM 帧字幕生成，例如静音视频、音乐视频或语音极少的视频。
6. **定向视觉查找**：在问答模式中，如果问题需要视觉细节（例如“黑板上写了什么公式？”），只采样相关时间戳的帧进行描述，而不是处理整段视频。

这个设计让视频理解更聪明也更高效。带字幕的 30 分钟课程视频可以在不调用 MLLM 的情况下完成分析。

## 快速开始

### 1. 安装

```bash
# 轻量 CLI/API 安装
pip install -e .

# 系统依赖：ffmpeg，Python 3.11+
```

可选功能组：

```bash
# ASR 兜底、OCR、情绪分析、实时视频和本地服务辅助脚本
pip install -e ".[asr,ocr,emotion,live,serving]"

# 本地开发的完整依赖安装
pip install -r requirements-full.txt
```

### 1.5 Hermes

本仓库内置 Hermes 原生 skill，路径为 `.agents/skills/media/vidify`，因此 Hermes 可以直接从当前 checkout 使用 Vidify。

将它安装到用户级 Hermes skills 目录：

```bash
python -m agent.main hermes install-skill
```

默认会以 symlink 方式安装到 `~/.hermes/skills/media/vidify`。如果希望安装为独立副本，可以使用 `--strategy copy`。

### 2. 可选运行环境

```bash
cp .env.example .env
# 按需编辑 .env，配置本地模型端点、模型名称和缓存路径。
```

### 3. 启动模型服务

**Qwen3.5（推荐）：**

```bash
# Qwen3.5 需要 vLLM >= 0.19.0
pip install "vllm>=0.19.0"

# 自动检测本地模型，或从 HuggingFace 下载
bash scripts/serving_qwen3_5.sh

# 也可以手动启动：
vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --allowed-local-media-path $(pwd)/cache
```

**使用已有 GPU 端点验证：**

```bash
bash scripts/run_test_gpu.sh --api-base http://localhost:8000/v1 --video media/my_video.mp4
```

**Qwen3-VL（legacy）：**

```bash
bash scripts/serving_qwen3vl.sh
```

多 GPU 主机：

```bash
TP_SIZE=2 MAX_MODEL_LEN=131072 bash scripts/serving_qwen3_5.sh
```

### 4. 运行

**CLI：**

```bash
python -m agent.main analyze youtube "https://www.youtube.com/watch?v=..." --mode detailed

# 使用结构化 JSON 日志
python -m agent.main --log-format json analyze youtube "https://www.youtube.com/watch?v=..." --mode detailed

# 安装 Hermes skill 到 ~/.hermes/skills/media/vidify
python -m agent.main hermes install-skill
```

**REST API：**

```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000

# 标准接口（返回最终 JSON）
curl -X POST http://localhost:9000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"youtube", "uri":"https://www.youtube.com/watch?v=...", "mode":"detailed"}'

# 流式接口（通过 Server-Sent Events 返回实时进度）
curl -N -X POST http://localhost:9000/analyze/stream \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"youtube", "uri":"https://www.youtube.com/watch?v=...", "mode":"detailed"}'
```

**Web GUI：**

启动服务后打开 `http://localhost:9000`。

## 工作流模式

`brief` 是标准轻量模式。`quick` 仍作为兼容旧版本的别名，可在 CLI 和 API 中使用。

```bash
# 简要总结（ASR 优先，如果转录文本足够则跳过视觉处理）
python -m agent.main analyze youtube URL --mode brief

# 完整分析（OCR、情绪、目标、ASR、翻译）
python -m agent.main analyze youtube URL --mode detailed

# 即使转录文本足够，也强制运行视觉处理
python -m agent.main analyze youtube URL --mode brief --force-visual

# 构建搜索索引，然后提问
python -m agent.main analyze youtube URL --mode ask --question "What are the key conclusions?"

# 提出视觉问题（触发定向帧查找）
python -m agent.main analyze youtube URL --mode ask --question "What equation is shown on the board at 5:30?"

# 导出精彩片段
python -m agent.main analyze youtube URL --mode highlights

# 结合 Web 搜索生成报告
python -m agent.main analyze youtube URL --mode report --include-web-search

# 从摄像头实时处理
python -m agent.main analyze local webcam --mode live

# 从 RTMP/HTTP URL 实时处理
python -m agent.main analyze local stream --mode live --stream-source stream --stream-url rtmp://host/live/key
```

## Hermes

Vidify 通过两种方式支持 Hermes：

1. 通过 `.agents/skills/media/vidify` 提供原生 skill 集成
2. 通过 `agent.integrations.hermes` 提供稳定的 Python helper

Hermes wrapper 会优先使用已安装的 `vidify` CLI，也会回退到当前仓库里的 `python -m agent.main`，因此源码 checkout 也能直接使用。

如果你正在迁移较早的 OpenClaw setup，本仓库仍保留 `openclaw/` skill。

### 处理流程

```text
                    ┌─────────────┐
                    │  Download   │
                    │  + Metadata │ ← yt-dlp extracts info.json, subtitles
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Probe     │ ← ffprobe: duration, fps, resolution
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │  Subtitles available?    │
              └──┬──────────────────┬───┘
                 │ yes              │ no
          ┌──────▼──────┐   ┌──────▼──────┐
          │ Parse subs  │   │ Whisper ASR │
          └──────┬──────┘   └──────┬──────┘
                 └────────┬────────┘
                          │
                   ┌──────▼──────┐
                   │ Sufficiency │ ← coverage ≥ 30%? words ≥ 50?
                   │   check     │
                   └──┬──────┬───┘
                      │      │
            sufficient│      │ insufficient
                      │      │
               ┌──────▼──┐ ┌─▼──────────┐
               │  Skip   │ │ MLLM frame │
               │  MLLM   │ │ captioning │
               └──────┬──┘ └─┬──────────┘
                      └───┬──┘
                          │
                   ┌──────▼──────┐
                   │  Timeline   │ ← uses transcript + metadata + frames (if any)
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │    Save     │
                   └─────────────┘
```

## 在线 / 流式处理

Vidify 支持从摄像头和 RTMP/HTTP 流进行实时视频理解。流式架构参考了 [InternLM-XComposer-2.5-OmniLive](https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-2.5-OmniLive)。

### 架构

流式 pipeline 使用三模块设计：

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Live Stream Pipeline                        │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐   │
│  │  Perception   │   │     Memory       │   │    Reasoning      │   │
│  │              │   │                  │   │                   │   │
│  │ Frame capture│──▶│ Local segments   │──▶│ Query retrieval   │   │
│  │ Scene detect │   │ Global summary   │   │ LLM Q&A          │   │
│  │ SlowFast     │   │ Backup-on-query  │   │ Context building  │   │
│  └──────────────┘   └──────────────────┘   └───────────────────┘   │
│       │                                            │               │
│       ▼                                            ▼               │
│  Frame-level results                     Answer + evidence         │
└─────────────────────────────────────────────────────────────────────┘
```

**模块 A：感知**：按配置 FPS 捕获帧，使用 CLIP embedding 相似度检测场景变化（基于阈值，而不是固定窗口），并将每帧路由到重模型或轻模型分析（SlowFast 策略）。

**模块 B：记忆**：维护两级记忆结构：

- *局部记忆*：每个片段的压缩表示（caption + CLIP embedding），用于细粒度时间检索
- *全局记忆*：由 LLM 生成的跨片段摘要，用于整体理解

**模块 C：推理**：查询时会快照当前记忆（backup-on-query，保证一致性），通过余弦相似度检索相关片段，并使用完整上下文生成回答。

### 关键特性

| 特性 | 说明 |
|------|------|
| **自适应分段** | 基于 CLIP 的场景变化检测，生成语义上有意义的片段，而不是固定时长窗口 |
| **SlowFast 分析** | 每 N 帧使用重模型（7B MLLM + OCR + 检测），其他帧使用轻模型（小 MLLM + OCR） |
| **两级记忆** | 局部片段记忆用于检索，全局摘要用于整体理解 |
| **实时问答** | 可在流处理过程中提问；系统会快照记忆，保证检索一致 |
| **Backup-on-query** | 深拷贝记忆状态，确保检索过程基于一致数据 |

### CLI 用法

```bash
# 默认从摄像头读取实时流
python -m agent.main analyze local webcam --mode live

# RTMP 流
python -m agent.main analyze local stream --mode live \
  --stream-source stream --stream-url rtmp://host/live/key

# HTTP 流，例如 IP camera
python -m agent.main analyze local stream --mode live \
  --stream-source stream --stream-url http://camera-ip/video
```

### REST API

启动服务后使用 `/live/*` 端点：

```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000
```

**启动 session：**

```bash
curl -X POST http://localhost:9000/live/start \
  -H 'Content-Type: application/json' \
  -d '{"source": "stream", "stream_url": "rtmp://host/live/key", "fps": 1}'
# Returns: {"session_id": "live_0001", "status": "started"}
```

**流处理中提问：**

```bash
curl -X POST http://localhost:9000/live/ask \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "live_0001", "question": "What is happening in the video?"}'
# Returns: {"answer": "...", "relevant_segments": [...], "global_summary": "..."}
```

**查看状态：**

```bash
curl http://localhost:9000/live/status/live_0001
# Returns: {"running": true, "segments_processed": 12, "total_duration_sec": 180.0, ...}
```

**停止并获取最终记忆：**

```bash
curl -X POST http://localhost:9000/live/stop \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "live_0001"}'
# Returns: {"memory": {...}, "total_frame_results": 180}
```

### 流式配置

`workflows.yaml` 中的 `live_stream` 配置：

```yaml
live_stream:
  source: webcam               # "webcam" or "stream"
  fps: 1                       # frames per second to process
  heavy_interval: 5            # use heavy model every N frames
  similarity_threshold: 0.9    # CLIP cosine similarity for scene change
  min_segment_frames: 3        # minimum frames before allowing new segment
  max_segment_frames: 16       # force new segment after this many frames
```

`models.yaml` 中的模型层级：

```yaml
mllm:
  heavy:
    model_name: qwen3.5-9b     # Qwen3.5 unified VL model (recommended)
    base_url: http://localhost:8000/v1
  light:
    model_name: qwen3.5-4b     # Lightweight model for fast per-frame captioning
    base_url: http://localhost:8000/v1
```

## 端到端测试

### GPU 端点

`run_test_gpu.sh` 会对已有的 GPU-backed OpenAI-compatible endpoint 执行验证。

```bash
# 使用已经运行的 vLLM 端点
bash scripts/run_test_gpu.sh --api-base http://localhost:8000/v1 --video media/my_video.mp4

# 只运行指定测试
bash scripts/run_test_gpu.sh --api-base http://localhost:8000/v1 \
  --video media/my_video.mp4 --tests "frame_caption video_qa highlights"
```

### 本地 / 手动测试

`test_all.py` 会针对本地视频运行全部 17 个 skill 测试：

```bash
# 自动检测/启动 serving，并运行全部测试
python scripts/test_all.py --video-path media/taste_in_china_s1e1.mp4

# 使用已有端点
python scripts/test_all.py --video-path media/taste_in_china_s1e1.mp4 --api-base http://localhost:8000/v1

# 指定测试
python scripts/test_all.py --video-path media/taste_in_china_s1e1.mp4 --tests frames qa highlights
```

测试项：`video_probe` | `frame_sample` | `audio_extract` | `asr` | `ocr` | `object_detection` | `subtitle_parse` | `metadata_extract` | `content_sufficiency` | `needs_visual` | `asr_first_brief` | `frame_caption` | `video_caption` | `timeline` | `video_qa` | `highlights` | `video_edit`

### YouTube E2E

`test_youtube_e2e.py` 会自动发现或启动模型服务，下载 YouTube 视频，并运行完整测试套件：

```bash
# 自动检测/启动 serving，并运行全部测试
python scripts/test_youtube_e2e.py

# 使用已有端点
python scripts/test_youtube_e2e.py --api-base http://localhost:8000/v1

# 自定义视频和指定测试
python scripts/test_youtube_e2e.py \
    --youtube "https://www.youtube.com/watch?v=..." \
    --tests frames qa multi_turn_qa
```

测试项：`frames` | `batch_frames` | `video_caption` | `qa` | `multi_turn_qa`

完整说明见 [Testing Guide](docs/testing.md)。

## 生产特性

Vidify 包含面向生产环境的强化模式，借鉴了大规模 agent 架构中的实践。

### 指数退避重试

所有模型调用（vLLM chat、Whisper ASR、embedding API）都会对瞬时失败（超时、连接错误、5xx、限流）自动重试。每次调用可配置 `max_retries`、`base_delay`、`max_delay`，并带 jitter，避免重试风暴。

```python
from agent.core.retry import retry_with_backoff

@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0)
def my_api_call():
    ...
```

### 优雅降级

可选 skill（OCR、目标检测、情绪分析、翻译、Web 搜索）会通过 `@skill_guard` 包裹。如果依赖缺失或模型失败，对应 skill 会被跳过，pipeline 继续执行并给出 warning，而不是直接崩溃。

### 并行 skill 执行

在 `detailed` 工作流中，彼此独立的 skill（OCR、目标检测、情绪分析）会通过线程池并行运行。可在 `workflows.yaml` 中通过 `max_parallel_skills` 配置，默认值为 3。

### 并行分段处理

对长视频（默认大于 5 分钟），`brief` 和 `detailed` 工作流都可以将视频切分为时间片段，并发处理后再合并：

```text
Long Video → split into N segments (by duration)
                ↓
    ┌───────────┼───────────┐
    Seg 0       Seg 1       Seg 2  ...  (parallel workers)
    │           │           │
    frames      frames      frames
    caption     caption     caption
    OCR         OCR         OCR
    detection   detection   detection
    emotion     emotion     emotion
    └───────────┼───────────┘
                ↓
         Merge results (adjust timestamps)
                ↓
         Timeline builder (on merged data)
```

**保持全局处理的部分**：probe、充分性检查、timeline、translation、web search。
**可并行处理的部分**：frame sampling、MLLM captioning、OCR、object detection、emotion analysis。
**也可并行的部分**：长音频 Whisper ASR，可通过音频切片/合并并调整时间戳。

在 `workflows.yaml` 中启用：

```yaml
detailed:
  parallel_segments:
    enabled: true              # activate parallel processing
    segment_duration: 300      # seconds per segment (5 min)
    max_workers: 4             # concurrent segment workers
    min_video_duration: 300    # only for videos longer than this
    min_segment_duration: 30   # merge tiny tail into previous segment
  parallel_asr:
    enabled: true
    max_workers: 4
    segment_duration: 240      # seconds per ASR clip
    min_audio_duration: 300
    min_segment_duration: 30
```

**可插拔分段**：分段策略通过 `BaseSegmentor` 接口抽象（`agent/core/segment.py`）。默认 `DurationSegmentor` 使用 FFmpeg 按固定时长切分。自定义分段器（例如基于 TransNetV2 的时序边界检测，或基于 CLIP 的语义分段）可以在运行时注册：

```python
from agent.core.segment import BaseSegmentor, register_segmentor

class SceneSegmentor(BaseSegmentor):
    """DL-based scene boundary detection (e.g., TransNetV2)."""
    def segment(self, video_path, duration_sec, base_cache_dir):
        boundaries = my_model.predict(video_path)  # your model here
        segments = []
        for i, (start, end) in enumerate(boundaries):
            segments.append(self._make_segment(i, start, end, base_cache_dir))
        return self._merge_tiny_tail(segments, duration_sec)

register_segmentor("scene", SceneSegmentor)
# Then set segmentor_name="scene" in config or split_video_into_segments()
```

### 流式进度事件

事件总线（`agent.core.events`）会在 pipeline 每一步发出生命周期事件：`skill_start`、`skill_complete`、`skill_error`、`skill_skipped`、`progress`。

- **CLI**：实时将每个 skill 的进度输出到 stderr
- **API**：`POST /analyze/stream` 返回 Server-Sent Events，用于监控实时进度

### 生命周期钩子

可以通过 `hooks.yaml` 在分析关键节点触发 shell 命令：

```yaml
hooks:
  post_analysis:
    - command: "curl -X POST $WEBHOOK_URL -d @$RESULT_PATH"
      async: true
      timeout: 10
  on_error:
    - command: "echo 'Failed: $ERROR_MSG' >> errors.log"
```

钩子点：`pre_analysis`、`post_analysis`、`post_skill`、`on_error`、`post_highlight`、`post_index`。

### 结构化日志

CLI 可传入 `--log-format json`，输出机器可读的 JSON 日志，包含 `video_id`、`skill_name`、`duration_ms` 和 `status` 等字段。也可以使用 `WorkflowTracker` 汇总每个工作流中的 skill 耗时。

## 项目结构

```text
agent/
  core/
    schemas.py           # Data models (VideoAsset, FrameSet, Transcript, ContentMetadata, ...)
    orchestrator.py      # Workflow router with hook triggers
    segment.py           # Parallel segment processing: BaseSegmentor interface, DurationSegmentor, merge functions
    segment_worker.py    # Per-segment pipeline worker (frames → caption → OCR/detection/emotion)
    retry.py             # @retry_with_backoff decorator (exponential backoff + jitter)
    skill_guard.py       # @skill_guard decorator (graceful degradation)
    events.py            # EventBus for streaming progress notifications
    parallel.py          # Parallel execution: run_skills_parallel + run_segments_parallel
    hooks.py             # Lifecycle hook manager (reads hooks.yaml)
    logging_config.py    # Structured JSON logging, WorkflowTracker
  extensions/
    models/              # vLLM client, direct model loader
      thinking.py          # Qwen3.5 thinking mode utilities (strip/extract/disable)
    skills/              # Processing skills
      subtitle_parser.py   # VTT/SRT parsing into Transcript
      content_sufficiency.py # Heuristic check: skip visuals if transcript is enough
      video_download.py    # yt-dlp with metadata + subtitle extraction
      asr.py               # Whisper ASR
      vision_caption.py    # MLLM frame/video captioning
      timeline_builder.py  # LLM-based timeline (uses content metadata)
      scene_similarity.py  # CLIP-based scene-change detection for streaming
      stream_memory.py     # Two-level memory manager (local + global)
      live_stream_processing.py # Real-time stream processor with SlowFast
      ...                  # OCR, object detection, emotion, FAISS, etc.
    workflows/           # Pipelines (brief, detailed, index, ask, highlights, report, live)
    utils/               # Caching, hashing, serving utilities
      serving.py           # vLLM discovery, launch (Qwen3.5/3-VL), health monitoring
  config.py              # YAML config loader
  main.py                # CLI entry point
server/
  app.py                 # FastAPI REST server (port 9000)
scripts/                 # Test and demo scripts
  run_test_gpu.sh          # Run tests against a GPU-backed endpoint
  run_test_ascend.sh       # Run tests against an Ascend/NPU-backed endpoint
  test_all.py              # 17-skill test suite for local videos
  test_youtube_e2e.py      # YouTube E2E test
  serving_qwen3_5.sh       # vLLM serving for Qwen3.5-9B (GPU)
  serving_qwen3vl.sh       # vLLM serving for Qwen3-VL (legacy)
  serving_qwen2_5vl_ascend.sh # vLLM serving for Qwen2.5-VL on Ascend/NPU
  serving_qwen3_5_ascend.sh   # vLLM serving for Qwen3.5-9B on Ascend/NPU
  start_vidify_ascend.sh  # Convenience wrapper for Ascend/NPU serving
docs/                    # Detailed documentation
.env                     # Optional local runtime overrides (gitignored)
.env.example             # Template for .env
```

## 配置

### 运行环境（`.env`）

本地 override 可以写入 `.env`（该文件已 gitignore）：

```bash
cp .env.example .env
```

| 变量 | 说明 | 示例 |
|------|------|------|
| `LLM_BASE_URL` | OpenAI-compatible chat/completions endpoint | `http://localhost:8000/v1` |
| `LLM_MODEL` | 默认多模态模型名称 | `qwen3.5-9b` |
| `EMBED_BASE_URL` | OpenAI-compatible embeddings endpoint | `http://localhost:8000/v1` |
| `EMBED_MODEL` | 默认 embedding 模型名称 | `qwen-embed` |
| `CACHE_ROOT` | 运行时缓存目录 | `./cache` |

### 模型与工作流配置

项目根目录中的可选 YAML 文件：

- `models.yaml`：模型选择、参数、端点
- `workflows.yaml`：工作流步骤、帧数限制、功能开关
- `hooks.yaml`：生命周期钩子，在分析关键节点触发 shell 命令

### ASR 优先配置

`workflows.yaml` 中的这些设置控制 ASR-first 行为：

```yaml
brief:
  asr_first: true                    # enable ASR-first mode (default: true)
  min_coverage_ratio: 0.3            # minimum speech-to-video duration ratio
  min_word_count: 50                 # minimum transcript words to be "sufficient"
  force_visual: false                # override: always run MLLM captioning
  prefer_subtitles_over_asr: true    # use embedded subs over Whisper when available
```

如果配置文件不存在，会回退到内置默认值。CLI/API 参数始终优先。

更多内容见 [Configuration Guide](docs/configuration.md)。

## Ascend / NPU 部署

Vidify 可以通过与 GPU serving 相同的 OpenAI-compatible API 对接 Ascend-backed vLLM 部署。项目提供了面向常见 Qwen 模型的通用 helper 脚本：

公开文档中的部署示例应保持通用。请将具体提供商环境名称、内部 registry URL、挂载路径和调度命令保存在本地文档或 `.env` 文件中，不要提交到 README。

```bash
# Qwen3.5-9B
TP_SIZE=2 bash scripts/serving_qwen3_5_ascend.sh /models/Qwen3.5-9B

# Qwen2.5-VL fallback
TP_SIZE=2 bash scripts/serving_qwen2_5vl_ascend.sh /models/Qwen2.5-VL-7B-Instruct

# Run tests against an existing Ascend/NPU endpoint
bash scripts/run_test_ascend.sh --api-base http://localhost:8000/v1 --video media/my_video.mp4
```

对 Ascend 上的 Qwen3.5，helper 默认使用 `--enforce-eager` 和较保守的 `MAX_MODEL_LEN=16384`，通常更适合 NPU 显存行为。请根据你的硬件和 vLLM 构建调整 `TP_SIZE`、`MAX_MODEL_LEN`、`PORT` 和 `ALLOWED_LOCAL_MEDIA_PATH`。

## Docker

```bash
# Full stack (app + vLLM)
docker-compose up

# App only
docker build -t vidify .
docker run -p 9000:9000 vidify
```

## 依赖要求

- **系统**：ffmpeg、yt-dlp、Python 3.11+
- **GPU**：用于模型服务的 vLLM-compatible GPU，或使用 `--direct-model` 进行本地加载
- **模型**：Qwen3.5（默认，推荐）、Qwen3-VL（legacy），可通过 `models.yaml` 配置
- **vLLM**：Qwen3.5 需要 >= 0.19.0（`pip install "vllm>=0.19.0"`）

## 文档

| 文档 | 内容 |
|------|------|
| [Architecture](docs/architecture.md) | 数据模型、数据流、缓存结构、模型接口 |
| [Workflows](docs/workflows.md) | 所有工作流 pipeline 的详细说明 |
| [Skills Reference](docs/skills.md) | 所有 skill 的 API 签名和说明 |
| [API Reference](docs/api.md) | REST endpoint、CLI option、请求/响应 schema |
| [Configuration](docs/configuration.md) | YAML 配置、vLLM setup、Docker 部署 |
| [Testing Guide](docs/testing.md) | E2E 测试脚本、demo 脚本、单项测试 |
| [Web Search](docs/web-search.md) | Google/Baidu 搜索集成设置 |
