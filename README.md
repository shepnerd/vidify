# VidCopilot

一个视频处理/分析/理解 Agent：
- 输入：YouTube URL（优先），后续可扩展任意 URL、本地视频、直播流
- 处理：下载缓存、ffmpeg 解码、场景切分抽关键帧（默认<=128）、音频提取、ASR
- 理解：关键帧 caption、多段结构化时间线（chapters/events）、OCR、情感分析、对象检测
- 检索：帧描述+ASR+元数据一起入库，FAISS 检索 + LLM 回答
- 编辑：高光检测，支持 clips 分段导出 + reel 合集拼接
- **深度搜索**：集成 Google Custom Search API，提供网络上下文增强
- **报告生成**：整合多源分析结果，生成综合视频分析报告
- **新增技能**：OCR（文本提取）、情感分析、对象检测、翻译、批量处理、自定义摘要、实时流处理（支持摄像头，SlowFast策略，轻量模型）

## 1. 依赖
- 本机：ffmpeg、Python 3.11+
- Python 依赖：见 requirements.txt
- 下载：yt-dlp

## 2. 启动 vLLM（示例）
你需要单独启动 vLLM OpenAI-compatible server（提供 /v1/chat/completions 与 /v1/embeddings）。
关键注意点：
- vLLM 默认会应用模型仓库的 generation_config.json；如需禁用，启动时加 `--generation-config vllm`。
- Chat Completions 需要模型 tokenizer 有 chat template；没有的话需要 `--chat-template ...` 指定，否则所有 chat 请求会报错。
- 本项目使用本地帧路径作为 image_url，因此 vLLM 需要 `--allowed-local-media-path` 放行 cache 目录。

示例（请替换模型路径/名称）：
```bash
vllm serve /models/qwen-vl \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen-vl \
  --generation-config vllm \
  --allowed-local-media-path /abs/path/to/cache
```

## 3.5 深度搜索和报告生成功能

VidCopilot 支持通过 Google Custom Search API 增强视频分析能力，并生成综合分析报告。

### 功能特点
- **深度搜索增强**: 基于视频内容自动生成搜索查询，从网络获取相关上下文信息
- **智能报告生成**: 整合视频元数据、时间线、关键帧、转录内容和网络搜索结果
- **多地区支持**: 自动检测网络环境，支持Google搜索和百度搜索，适配中国用户
- **灵活配置**: 支持环境变量或命令行参数配置API凭证
- **向后兼容**: 不使用时不影响现有功能

### 设置 Google Custom Search API

1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 创建项目并启用 Custom Search API
3. 创建 API 密钥和自定义搜索引擎
4. 设置环境变量：
```bash
export GOOGLE_API_KEY="your_api_key_here"
export GOOGLE_SEARCH_ENGINE_ID="your_search_engine_id_here"
```

详细设置请参考 [GOOGLE_SEARCH_SETUP.md](GOOGLE_SEARCH_SETUP.md)

### 中国用户支持
VidCopilot 自动检测网络环境，对中国用户提供以下支持：

- **自动切换**: 当无法访问Google服务时，自动切换到百度搜索
- **本地回退**: 在网络受限情况下，提供本地搜索建议和使用指南
- **无缝体验**: 无需手动配置，系统自动选择最佳的搜索服务

**网络检测逻辑**:
1. 优先尝试Google Custom Search (需要API密钥)
2. 如果Google不可用，自动切换到百度搜索
3. 如果所有外部搜索都不可用，提供本地帮助信息

### 命令行使用

#### 带深度搜索的简要分析
```bash
python agent/main.py --source-type youtube --uri "https://www.youtube.com/watch?v=XXXX" --mode brief --include-web-search
```

#### 带深度搜索的详细分析
```bash
python agent/main.py --source-type youtube --uri "https://www.youtube.com/watch?v=XXXX" --mode detailed --include-web-search
```

#### 生成综合分析报告
```bash
python agent/main.py --source-type youtube --uri "https://www.youtube.com/watch?v=XXXX" --mode report --analysis-type detailed --include-web-search
```

#### 直接指定API凭证
```bash
python agent/main.py --source-type youtube --uri "https://www.youtube.com/watch?v=XXXX" --mode brief --include-web-search --google-api-key "your_key" --google-search-engine-id "your_id"
```

### API用法

#### 带深度搜索的分析
```bash
curl -X POST http://localhost:9000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type":"youtube",
    "uri":"https://www.youtube.com/watch?v=XXXX",
    "mode":"detailed",
    "cache_root":"./cache",
    "llm_base_url":"http://localhost:8000/v1",
    "llm_model":"qwen-vl",
    "include_web_search": true,
    "google_api_key": "your_api_key",
    "google_search_engine_id": "your_search_engine_id"
  }'
```

#### 生成报告
```bash
curl -X POST http://localhost:9000/report \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type":"youtube",
    "uri":"https://www.youtube.com/watch?v=XXXX",
    "cache_root":"./cache",
    "analysis_type": "detailed",
    "include_web_search": true,
    "llm_base_url":"http://localhost:8000/v1",
    "llm_model":"qwen-vl",
    "google_api_key": "your_api_key",
    "google_search_engine_id": "your_search_engine_id"
  }'
```

### 测试深度搜索功能
```bash
python test_web_search.py
```

### 多地区搜索演示
```bash
python demo_multi_region_search.py
```

此演示脚本会自动检测您的网络环境并展示相应的搜索功能。

## 安装指南

### 一键安装
运行 `./setup.sh` 自动安装依赖和配置环境。

### 手动安装
1. 安装系统依赖：ffmpeg, Python 3.11+
2. `pip install -r requirements.txt`
3. 下载模型并配置环境变量。

### Docker
```bash
docker build -t vidcopilot .
docker run -p 9000:9000 vidcopilot
```

## 配置管理

VidCopilot 支持通过 YAML 配置文件进行模型和流程的集中管理，避免硬编码并便于自定义。

### 模型配置 (models.yaml)
```yaml
# models.yaml - 模型配置
mllm:
  heavy:
    model_name: "qwen-vl-7b"
    base_url: "http://localhost:8000/v1"
    max_tokens: 512
    temperature: 0.7
  light:
    model_name: "qwen-vl-1b"
    base_url: "http://localhost:8000/v1"
    max_tokens: 256
    temperature: 0.5

ocr:
  engine: "paddleocr"
  lang: "ch"
  use_angle_cls: true

object_detection:
  model: "yolov8n.pt"
  conf_threshold: 0.5

asr:
  model: "whisper"
  size: "small"
  language: null

emotion_analysis:
  audio_model: "wav2vec2-emotion"
  visual_model: "fer"

translation:
  source_lang: "en"
  target_lang: "zh"
  model: "helsinki-nlp"
```

### 工作流配置 (workflows.yaml)
```yaml
# workflows.yaml - 工作流配置
brief:
  steps:
    - download_video
    - extract_audio
    - asr
    - vision_caption
    - summarize
  output_format: "text"

detailed:
  steps:
    - download_video
    - extract_audio
    - asr
    - vision_caption
    - ocr
    - object_detection
    - emotion_analysis
    - translation
    - timeline_build
    - summarize
  output_format: "json"

live_stream:
  steps:
    - process_live_stream
  heavy_interval: 30  # 每30帧使用重模型
  resolution: "640x480"
  fps: 10
  output_format: "stream"
```

### 配置使用
- 配置文件位于项目根目录：`models.yaml` 和 `workflows.yaml`
- 如果配置文件不存在，系统将使用内置默认配置
- 工作流函数会自动加载配置，如果未提供参数则使用配置文件中的默认值
- 可以覆盖特定参数以进行自定义

### Docker Compose
```bash
docker-compose up
```

### PyPI
```bash
pip install vidcopilot
vidcopilot analyze youtube https://... --mode detailed
```

## 入门教程
1. 启动vLLM服务器。
2. 运行 `python agent/main.py analyze youtube <URL> --mode detailed`。
3. 或访问 `http://localhost:9000` 使用GUI。

## 常见问题
- 模型加载失败：检查vLLM配置。
- 网络问题：配置代理或使用本地模式。

## 示例用例
- 分析YouTube视频：`vidcopilot analyze youtube <URL>`
- 批量处理：使用batch_processing技能。
- 实时流处理：`process_live_stream(source='webcam', callback=my_callback, resolution=(640,480), fps=1, heavy_interval=5)`

## 5. API用法
Analyze（detailed）
```bash
curl -X POST http://localhost:9000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type":"youtube",
    "uri":"https://www.youtube.com/watch?v=XXXX",
    "mode":"detailed",
    "cache_root":"./cache",
    "llm_base_url":"http://localhost:8000/v1",
    "llm_model":"qwen-vl",
    "max_frames":128,
    "whisper_model":"small"
  }'
```

Index（FAISS）
```bash
curl -X POST http://localhost:9000/index \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type":"youtube",
    "uri":"https://www.youtube.com/watch?v=XXXX",
    "cache_root":"./cache",
    "llm_base_url":"http://localhost:8000/v1",
    "llm_model":"qwen-vl",
    "embed_base_url":"http://localhost:8000/v1",
    "embed_model":"qwen-embed",
    "chunk_sec":20
  }'
```

Ask
```bash
curl -X POST http://localhost:9000/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type":"youtube",
    "uri":"https://www.youtube.com/watch?v=XXXX",
    "cache_root":"./cache",
    "question":"视频的核心观点是什么？分别对应哪些时间段？",
    "top_k":5,
    "llm_base_url":"http://localhost:8000/v1",
    "llm_model":"qwen-vl",
    "embed_base_url":"http://localhost:8000/v1",
    "embed_model":"qwen-embed"
  }'
```

Highlights（输出 clips + 可选 reel）
```bash
curl -X POST http://localhost:9000/highlights \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type":"youtube",
    "uri":"https://www.youtube.com/watch?v=XXXX",
    "cache_root":"./cache",
    "llm_base_url":"http://localhost:8000/v1",
    "llm_model":"qwen-vl",
    "max_clips":5,
    "also_make_reel":true
  }'
```

读取 analysis.json（给前端画时间线）
```bash
curl -X POST http://localhost:9000/analysis \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type":"youtube",
    "uri":"https://www.youtube.com/watch?v=XXXX",
    "cache_root":"./cache"
  }'
```

6. 端到端 Demo 脚本

## 架构图

```
VidCopilot Architecture
├── agent/
│   ├── core/          # 核心编排和模式
│   ├── extensions/    # 扩展技能和工作流
│   │   ├── skills/    # 技能模块 (ASR, OCR, etc.)
│   │   ├── workflows/ # 工作流 (analyze, index, etc.)
│   │   ├── models/    # 模型加载器
│   │   ├── storage/   # 存储和持久化
│   │   └── utils/     # 工具函数
│   ├── config.py      # 配置管理
│   └── main.py        # CLI入口
├── server/            # FastAPI服务器 (/docs for Swagger)
├── tests/             # 单元和集成测试
└── scripts/           # 演示脚本
```

## API文档

启动服务器后，访问 `http://localhost:9000/docs` 查看Swagger UI。