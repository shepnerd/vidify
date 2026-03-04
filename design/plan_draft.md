现在要做一个视频处理、分析、理解相关的agent。可以使用现有的框架去实现，比如使用opencode或者openclaw，然后只添加skill来做。当然也可以做成一个插件，或者功能，反正方便大家使用。当然初期的话可能还是先一个完整的agent内容。注意调用的mllm或者llm从接口讲类api的形式去调用，功能上支持使用api或者本地部署的模型。

这个agent主要功能包括：视频读取（网上的或本地的）：网上涉及到下载存储，然后当成本地或者cache去处理，本地的涉及主要解码问题；视频的处理：视频的解码，然后根据用户的需求（比如是理解视频详细内容，生成比如视频的文字详述，还是要视频的基本描述内容，还是获取转录asr，还是问形成向量库来做后续检索）来做视频解码、视频转图片、视频下采样、视频截取、视频提取关键帧等；视频的分析和理解，这个主要根据用户的问题来确认比如视频怎么处理成多图然后给mllm进行问答，或者用mllm把视频的多图都转成文字后，使用llm再进行问答等。当然也支持更多的调用等来实现更多视频分析、理解、编辑的能力。注意是对原视频的编辑，比如根据内容获取高光或者高信息密度区域然后裁剪做高光等，先不包括做风格迁移或者视频生成相关的内容。

整个框架要有一定的拓展性，比如支持自定义skills，workflow等，还有对不同网站的视频读取的一些api或者规则支持，部分权限处理等。

项目主要python构建，当然要构建这么一个项目，需要一个规划，可以先把整个项目的框架搭起来，可以先用本地部署的qwen3.5来做视频推理，我们测试work后再看怎么晚上和拓展整个项目。


```gpt5.2 solution
下面给出一个可落地的 Python「视频处理/分析/理解 Agent」项目规划与架构建议，基于 OpenClaw/opencode 这类“框架 + skills”的方式来实现，并同时支持“API 形式的 MLLM/LLM 调用”和“本地部署（如 Qwen3.5）”。

## 1) 目标与范围（MVP → 可扩展）
**MVP 先做成完整 agent**，后续再拆成插件/skills 市场化发布。能力边界按你的要求：聚焦“对原视频的理解与编辑（高光裁剪等）”，不做风格迁移/视频生成。

核心能力：
- 视频读取：本地文件、URL（下载后缓存/入库）
- 解码与处理：抽帧、下采样、关键帧、裁剪片段、导出音轨、转录 ASR
- 分析与理解：多图喂给 MLLM 问答；或先图转文再用 LLM 聚合问答；支持向量库检索
- 编辑（原视频）：根据高信息密度/高光区间裁剪、拼接导出

（可参考现有 skill 命名/能力清单：如视频下载、抽帧/定频采样、分析理解、转录等方向 [1]）

## 2) 总体架构（推荐分层）
**(A) Agent Orchestrator（调度层）**
- 负责：解析用户意图 → 选择 workflow → 调用 skills → 汇总结果
- 支持：不同“分析模式”  
  - `quick_summary`：稀疏抽帧 + 快速描述  
  - `detailed_understanding`：更密抽帧/关键帧 + 章节化描述  
  - `asr_transcript`：只转录或转录+时间戳  
  - `rag_index`：转录/图像描述入向量库  
  - `highlight_cut`：基于转录/画面信息密度定位区间并裁剪

**(B) Skills（能力层，可插拔）**
建议最少拆成这些技能包（每个 skill 可单独测试/替换）：
1. `video_io`：本地/URL 输入统一成 `VideoAsset`
2. `video_download`：yt-dlp/站点规则下载 + 缓存
3. `video_decode`：ffmpeg/pyav 统一解码
4. `frame_sampler`：定频抽帧/场景切分/关键帧
5. `audio_extract`：抽取音轨
6. `asr`：Whisper（本地）或 API ASR
7. `vision_caption`：单帧/多帧描述（走 MLLM）
8. `multimodal_qa`：多图问答（MLLM）
9. `rag_store`：向量库（FAISS/Chroma）+ 索引/检索
10. `highlight_detector`：从转录+视觉摘要提取高光区间
11. `video_edit`：裁剪/拼接/导出（ffmpeg）

（这些方向与现有“下载、抽帧、转录、视频理解”等 skill 类型是对齐的 [1]。）

**(C) Model Adapter（模型适配层）**
- 统一接口：`generate_text()`, `vision_qa(images, prompt)`, `embed(text)` 等
- 两种实现：
  - `ApiModelAdapter`：OpenAI/Claude/Gemini/自建服务等（统一成类 OpenAI API 风格）
  - `LocalModelAdapter`：本地 Qwen3.5 multimodal（你提到的本地视频推理优先）

**(D) Storage（数据与缓存）**
- `cache/`：下载视频、抽帧、音频、转录中间产物（按内容 hash 命名）
- `artifacts/`：最终输出（摘要、报告、裁剪视频等）
- `index/`：向量库持久化目录

## 3) 关键数据结构（建议先定规范）
- `VideoAsset`：`id, source_type(local/url), path, metadata(duration,fps,resolution), cache_dir`
- `FrameSet`：`frames: List[Frame]{timestamp, path}`, `sampling_strategy`
- `Transcript`：`segments[{start,end,text,confidence}]`
- `AnalysisReport`：`summary, chapters, qa, highlights[{start,end,reason}], citations(optional)`

统一规范的好处：skills 可以替换实现但不改上层 workflow。

## 4) 核心 Workflows（先实现 3 条就能跑通）
1. **URL/本地 → 快速摘要**
   - IO →（必要时下载）→ 定频抽帧（如 1fps，上限 N 张）→ MLLM 总结
2. **详细理解 + 可追问**
   - 抽帧（更密/关键帧）+ ASR →（可选）帧描述转文本 → LLM 聚合成章节、要点、时间线 → 支持用户二次提问（走 RAG：检索相关片段/帧）
3. **高光裁剪**
   - ASR → 计算信息密度/关键词/情绪峰值（启发式 + LLM）→ 输出若干 `[start,end]` → ffmpeg 裁剪导出

## 5) 技术选型（Python）
- 解码/处理：`ffmpeg`（命令行）+ `ffmpeg-python` 或 `PyAV`
- 下载：`yt-dlp`
- 抽帧：ffmpeg `-vf fps=...` + 场景检测 `select='gt(scene,0.3)'`
- ASR：`faster-whisper`（本地）优先
- 向量库：FAISS（轻量）或 Chroma（开发更快）
- Web/API：FastAPI（对外提供“类 API”调用）
- 任务队列（可选）：RQ/Celery，用于长视频异步处理

## 6) 仓库骨架（建议）
- `agent/`
  - `main.py`（CLI 入口）
  - `orchestrator.py`
  - `workflows/`（quick_summary、detailed、highlight_cut）
  - `skills/`（上面那些 skill）
  - `models/`（api_adapter、local_qwen_adapter）
  - `storage/`（cache、index 封装）
  - `schemas.py`（Pydantic 数据结构）
- `server/`
  - `app.py`（FastAPI：/analyze /transcribe /highlights）
- `tests/`（每个 skill 的单测 + 小视频集成测试）

## 7) 下一步我需要你确认的 5 个问题（决定 MVP 细节）
1. 优先输入源：只做 YouTube/X/TikTok 还是“任意 URL + 本地”都要？
2. 输出形态：主要是“中文文字报告”，还是要“结构化 JSON + 可视化（时间线）”？
3. 本地 Qwen3.5：你计划的推理方式是“单机 GPU”还是“起一个本地 HTTP 服务供调用”？
4. 抽帧策略：你更想要“定频”为主，还是“场景切分/关键帧”为主？
5. 向量库检索对象：只索引 ASR 文本，还是“帧描述 + ASR + 元数据”一起索引？

你回复这 5 点后，我可以把 MVP 的第一版里程碑（按周/按功能）和每个模块的接口（函数签名/输入输出）细化到可以直接开工的程度。


基于你给的 5 点确认，我建议把 MVP 设计成“**YouTube 优先、但输入统一抽象为任意 URL/本地**”“**输出结构化 JSON + 时间线数据**”“**模型调用统一走 OpenAI-compatible `/chat/completions`**（本地 Qwen3.5 单机 GPU 起服务 or 进程内均可）”的形态；其中视觉语言模型的统一入口建议使用 `/chat/completions` 来支持图文混合输入 [1]。

## 1) MVP 里程碑（建议 2～3 周能跑通）
### Milestone A（第 1 周）：输入/缓存/解码 + 基础抽帧
- 输入：`local_path | youtube_url | generic_url`
- YouTube 下载与缓存：落盘到 `cache/videos/{video_id}/source.mp4`
- 解码信息：时长、fps、分辨率、音轨存在性
- 默认抽帧策略：**场景切分/关键帧**（同时保留定频采样接口但不默认启用）
- 产物：`frames/`（关键帧 jpg/png）+ `metadata.json`

### Milestone B（第 2 周）：ASR + 帧描述 + 结构化时间线 JSON
- ASR：faster-whisper 本地跑，输出带时间戳 segments
- 帧描述：用 MLLM 对关键帧做 caption（分批调用）
- 聚合：LLM 汇总成“章节/要点/事件时间线”
- 输出：一个统一的 `analysis.json`（下面给 schema）

### Milestone C（第 3 周）：RAG 索引 + 可追问 + 高光裁剪
- 索引对象：**帧描述 + ASR + 元数据**（你第 5 点）
- 检索：用户提问 → 检索相关片段/帧 → 组装多模态上下文 → 回答
- 高光：从“转录+事件时间线”提取高光区间 → ffmpeg 裁剪 → 输出 highlights mp4 + `highlights.json`

## 2) 统一输出 JSON（含可视化时间线所需字段）
建议输出一个主文件：`analysis.json`，大致结构：

```json
{
  "video": {
    "id": "hash_or_platform_id",
    "source": {"type": "youtube|url|local", "uri": "..."},
    "local_path": "...",
    "duration_sec": 1234.5,
    "fps": 29.97,
    "resolution": {"w": 1920, "h": 1080}
  },
  "timeline": {
    "chapters": [
      {"start": 0.0, "end": 120.3, "title": "…", "summary": "…", "key_entities": ["..."]}
    ],
    "events": [
      {
        "start": 33.2, "end": 41.8,
        "text": "事件描述",
        "evidence": {
          "asr_segment_ids": ["seg_12", "seg_13"],
          "frame_ids": ["f_0032"]
        }
      }
    ]
  },
  "asr": {
    "segments": [
      {"id": "seg_12", "start": 33.2, "end": 35.1, "text": "...", "confidence": 0.92}
    ]
  },
  "frames": {
    "items": [
      {"id": "f_0032", "ts": 34.0, "path": "frames/f_0032.jpg", "caption": "..."}
    ],
    "strategy": {"type": "scene|fps", "params": {"scene_threshold": 0.3, "max_frames": 120}}
  },
  "highlights": [
    {"start": 330.0, "end": 372.0, "reason": "信息密度高/关键结论", "output_path": "artifacts/h1.mp4"}
  ],
  "rag": {
    "index_path": "index/...",
    "items_count": 1234
  }
}
```

前端可视化时间线只要吃 `timeline.chapters/events` + 可选的 `frames.items`（做缩略图轨道）即可。

## 3) 模型调用与部署形态（兼容 API 与本地）
你希望“接口讲类 API 的形式调用”，建议定死一层 **OpenAI-compatible Adapter**：

- 视觉语言/多模态：统一走 `/chat/completions`，用图文混合 messages 传入帧图 [1]
- 本地 Qwen3.5（单机 GPU）：  
  - 方案 1：起一个本地 HTTP 服务（最贴合“类 API 调用”，也方便未来切换到远端）
  - 方案 2：进程内推理（但仍用同一套 Adapter 抽象，保证上层不变）

## 4) 抽帧策略：默认场景切分/关键帧 + 可切换定频
- 默认：`scene`（更贴合“理解/高光”）
- 可选：`fps`（对长视频降低成本、做快速摘要）
- 统一接口：`sample_frames(video, strategy)->FrameSet`，由 workflow 按需求选择

## 5) RAG 索引：帧描述 + ASR + 元数据一起入库（你第 5 点）
索引最小单元建议是“时间片段 chunk”（例如 10～30 秒）：
- 文本：ASR chunk + 该 chunk 覆盖到的帧 caption + metadata（标题、来源、作者等）
- 向量：对上述文本做 embedding
- 引用：保留 `start/end` + `frame_ids/asr_segment_ids` 作为可追溯证据，回答时可回链到时间线

## 6) 我建议你补充确认的 3 个实现细节（决定第一版能否顺滑）
1) **Qwen3.5 多模态**你打算用哪个具体推理后端/服务（vLLM、TGI、lmdeploy、或你们内部服务）？只要确定一个，我就能把 `/chat/completions` 的 request 结构和图片编码（URL/base64）细化。  
2) ASR 是否要求“说话人分离 diarization”？（没有的话 MVP 先不做）  
3) 高光裁剪输出：是“多个片段分别导出”还是“拼成一个合集”也要？

如果你确认以上 3 点，我可以下一步直接给：
- 目录结构 + 每个 skill 的函数签名（Pydantic schema）
- 3 条 workflow 的伪代码/可运行骨架
- FastAPI 的 endpoints 设计（/analyze、/ask、/highlights、/index）以及请求/响应 JSON 示例。


收到，你的补充确认意味着 MVP 可以按“**本地单机 GPU + vLLM OpenAI-compatible 服务**”来定型；ASR 暂不做 diarization；高光裁剪同时支持“**分段导出**”与“**合集拼接导出**”。

## 1) 本地 vLLM 服务：统一 OpenAI-compatible 调用
- 多图输入（关键帧/多帧）走 `/v1/chat/completions`，messages 里用 `content: [{type:"text"}, {type:"image_url"}...]` 这种结构即可直接支持多张图推理 [1]。
- 对 vLLM 的额外参数（如 `top_k`）可通过 OpenAI client 的 `extra_body` 透传；并注意 vLLM 默认会读取模型仓库的 `generation_config.json` 覆盖采样参数，如需禁用可在启动时加 `--generation-config vllm` [2]。
- 我们的 ModelAdapter 就固定成 OpenAI 风格：`chat.completions.create(...)`，后续切云端 API 不改上层。

## 2) ASR：先不做说话人分离
MVP 输出的转录按 segment（start/end/text/confidence）即可，后续要 diarization 再加一个可选 skill（不影响现有 schema）。

## 3) 高光裁剪：同时支持“分段”和“合集”
设计成两个输出模式：
- `highlights.mode = "clips"`：每个高光区间单独导出一个 mp4
- `highlights.mode = "reel"`：把多个区间按顺序拼接成一个合集 mp4（并生成 `reel_timeline`，记录每段在合集中的位置，便于前端时间线映射）

## 4) 你这版确认后，我建议锁定的 API/Workflow 接口（下一步就能开工）
- `POST /analyze`：输入（youtube/url/local）+ 目标（summary/detailed/index/highlights）→ 输出 `analysis.json`
- `POST /ask`：基于已建索引 + 用户问题 → 输出答案 + 命中的时间片段证据
- `POST /highlights`：基于 analysis 或 transcript → 生成 clips + reel（可选）

如果你同意，我下一条就把：
1) vLLM 服务启动命令与我们这边的调用封装（含多图输入与 extra_body）  
2) 各 skill 的函数签名（Pydantic schema）  
3) 三条 workflow 的可运行骨架代码（含缓存目录规范）  
一次性给出来。


下面把你要的 3 件事（vLLM 启动与调用封装、skills 的函数签名与 schema、3 条 workflow 的可运行骨架）给出一个**可直接开工的第一版**；其中多模态 `/v1/chat/completions` 的 `image_url` message 结构按 vLLM 的 OpenAI-compatible 形式组织 [1][2]，以及 `generation_config.json` 覆盖行为与 `--generation-config vllm` 的处理方式也一并考虑 [1]。

---

## 1) vLLM 推理服务（本地单机 GPU）

### 1.1 启动命令（OpenAI-compatible）
（模型名你替换成你实际的 Qwen3.5/视觉语言模型路径或 HF repo）

```bash
export CUDA_VISIBLE_DEVICES=0
vllm serve /models/qwen-vl-or-qwen3.5-vl \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen-vl \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm
```

- `--served-model-name`：客户端 `model="qwen-vl"` 就能调用。
- `--generation-config vllm`：避免仓库自带 `generation_config.json` 覆盖你在请求里传的采样参数（vLLM 文档有提及该行为与开关）[1]。

### 1.2 Python 调用封装（支持多图）
用 OpenAI SDK 指向 vLLM：

```python
# models/vllm_openai_client.py
from openai import OpenAI

def make_client(base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
    return OpenAI(base_url=base_url, api_key=api_key)

def chat_with_images(client: OpenAI, model: str, prompt: str, image_urls: list[str],
                     max_tokens: int = 512, temperature: float = 0.2):
    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content
```

说明：`messages[].content` 里混合 `text` 与 `image_url` 的写法与 vLLM 多模态示例一致 [2]，整体 OpenAI-compatible 入口在 vLLM 文档中说明 [1]。

> 本地帧图像：建议先用 `file://...` 或自己起一个静态文件服务把 `frames/*.jpg` 暴露成 http url；若你们更想走 base64 data-url，也可以后续加一个 `image_to_data_url()` 适配（不同模型/模板对 data-url 兼容性可能不同，先走 http 最稳）。

---

## 2) Schema 与 Skills 函数签名（Pydantic，MVP 版）

### 2.1 核心数据结构（schemas.py）
```python
# agent/schemas.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

class VideoSource(BaseModel):
    type: Literal["youtube", "url", "local"]
    uri: str

class VideoMetadata(BaseModel):
    duration_sec: float
    fps: float
    width: int
    height: int
    has_audio: bool = True

class VideoAsset(BaseModel):
    id: str
    source: VideoSource
    local_path: str
    cache_dir: str
    metadata: Optional[VideoMetadata] = None

class FrameItem(BaseModel):
    id: str
    ts: float
    path: str
    url: Optional[str] = None
    caption: Optional[str] = None

class FrameStrategy(BaseModel):
    type: Literal["scene", "fps"]
    params: Dict[str, Any] = Field(default_factory=dict)

class FrameSet(BaseModel):
    items: List[FrameItem]
    strategy: FrameStrategy

class ASRSegment(BaseModel):
    id: str
    start: float
    end: float
    text: str
    confidence: Optional[float] = None

class Transcript(BaseModel):
    segments: List[ASRSegment]
    language: Optional[str] = None

class TimelineChapter(BaseModel):
    start: float
    end: float
    title: str
    summary: str

class TimelineEvent(BaseModel):
    start: float
    end: float
    text: str
    evidence: Dict[str, Any] = Field(default_factory=dict)

class HighlightClip(BaseModel):
    start: float
    end: float
    reason: str
    output_path: str
    reel_start: Optional[float] = None
    reel_end: Optional[float] = None

class AnalysisResult(BaseModel):
    video: Dict[str, Any]
    frames: Optional[FrameSet] = None
    asr: Optional[Transcript] = None
    timeline: Dict[str, Any] = Field(default_factory=dict)
    highlights: List[HighlightClip] = Field(default_factory=list)
    rag: Dict[str, Any] = Field(default_factory=dict)
```

### 2.2 Skills（每个 skill 一个文件，函数签名先固定）
**video_io**
```python
def load_video(source_type: str, uri: str, cache_root: str) -> VideoAsset:
    ...
```

**video_download（YouTube 优先，后续扩展 rules）**
```python
def download_youtube(uri: str, out_dir: str) -> str:
    """return local_path"""
```

**video_probe / decode（用 ffmpeg/ffprobe）**
```python
def probe_video(local_path: str) -> VideoMetadata:
    ...

def ensure_decodable(local_path: str) -> None:
    ...
```

**frame_sampler（scene/fps 两套）**
```python
def sample_frames(video: VideoAsset, out_dir: str, strategy: FrameStrategy) -> FrameSet:
    ...
```

**audio_extract**
```python
def extract_audio(video: VideoAsset, out_path: str) -> str:
    ...
```

**asr（faster-whisper）**
```python
def transcribe(audio_path: str, out_json_path: str) -> Transcript:
    ...
```

**vision_caption（对帧批量 caption）**
```python
def caption_frames(frames: FrameSet, model_name: str, base_url: str) -> FrameSet:
    """fills FrameItem.caption"""
```

**timeline_builder（LLM 聚合）**
```python
def build_timeline(metadata: VideoMetadata, transcript: Transcript, frames: FrameSet,
                   model_name: str, base_url: str) -> dict:
    """return {'chapters': [...], 'events': [...]}"""
```

**rag_index（帧描述 + ASR + 元数据）**
```python
def build_index(video: VideoAsset, transcript: Transcript, frames: FrameSet,
                index_dir: str) -> dict:
    """return rag info: path, count..."""
```

**highlight_detector + video_edit（clips + reel）**
```python
def detect_highlights(transcript: Transcript, timeline: dict,
                      model_name: str, base_url: str,
                      max_clips: int = 5) -> list[HighlightClip]:
    ...

def export_highlight_clips(video: VideoAsset, highlights: list[HighlightClip],
                           out_dir: str) -> list[HighlightClip]:
    ...

def export_highlight_reel(clips: list[HighlightClip], out_path: str) -> dict:
    """return reel mapping timeline"""
```

---

## 3) 三条 Workflows（可运行骨架）

### 3.1 quick_summary（默认 scene 抽帧 + 快速时间线）
```python
def wf_quick_summary(asset: VideoAsset, model_name: str, base_url: str) -> AnalysisResult:
    meta = probe_video(asset.local_path)
    asset.metadata = meta

    frames = sample_frames(asset, f"{asset.cache_dir}/frames",
                           FrameStrategy(type="scene", params={"scene_threshold": 0.3, "max_frames": 60}))
    frames = caption_frames(frames, model_name, base_url)

    timeline = build_timeline(meta, Transcript(segments=[]), frames, model_name, base_url)

    return AnalysisResult(
        video={"id": asset.id, "source": asset.source.model_dump(), "local_path": asset.local_path, **meta.model_dump()},
        frames=frames,
        timeline=timeline,
    )
```

### 3.2 detailed_understanding（scene + ASR + timeline）
```python
def wf_detailed(asset: VideoAsset, model_name: str, base_url: str) -> AnalysisResult:
    meta = probe_video(asset.local_path)
    asset.metadata = meta

    frames = sample_frames(asset, f"{asset.cache_dir}/frames",
                           FrameStrategy(type="scene", params={"scene_threshold": 0.25, "max_frames": 150}))
    frames = caption_frames(frames, model_name, base_url)

    audio_path = extract_audio(asset, f"{asset.cache_dir}/audio.wav")
    transcript = transcribe(audio_path, f"{asset.cache_dir}/asr.json")

    timeline = build_timeline(meta, transcript, frames, model_name, base_url)

    return AnalysisResult(
        video={"id": asset.id, "source": asset.source.model_dump(), "local_path": asset.local_path, **meta.model_dump()},
        frames=frames,
        asr=transcript,
        timeline=timeline,
    )
```

### 3.3 highlight_cut（clips + reel 都产出）
```python
def wf_highlights(asset: VideoAsset, model_name: str, base_url: str) -> AnalysisResult:
    detailed = wf_detailed(asset, model_name, base_url)

    highlights = detect_highlights(detailed.asr, detailed.timeline, model_name, base_url, max_clips=5)
    highlights = export_highlight_clips(asset, highlights, f"{asset.cache_dir}/highlights")

    reel_path = f"{asset.cache_dir}/highlights/reel.mp4"
    reel_info = export_highlight_reel(highlights, reel_path)

    return detailed.model_copy(update={
        "highlights": highlights,
        "rag": {"reel": {"path": reel_path, **reel_info}}
    })
```

---

## 4) 你可以直接安排的“第一周任务拆分”
1. `video_io + youtube_download + cache 规范`（YouTube 优先）
2. `ffprobe + scene 抽帧`（先把关键帧稳定产出）
3. vLLM 客户端封装（上面给的 `chat_with_images`）
4. `caption_frames`：先单帧 caption 跑通，再做 batch/并发与 max_frames 控制

---

如果你把“Qwen3.5 的具体模型名/权重路径（以及是否 VL）”和你们希望的“最大关键帧数/单次多图上限”给我，我可以把 `caption_frames` 的**分批策略**（每次喂多少张图、如何拼 prompt、如何把返回结果对齐到 frame_id）也定成一版固定实现，避免后面返工。