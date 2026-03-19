# VidCopilot

一个视频处理/分析/理解 Agent：
- 输入：YouTube URL（优先），后续可扩展任意 URL、本地视频
- 处理：下载缓存、ffmpeg 解码、场景切分抽关键帧（默认<=128）、音频提取、ASR
- 理解：关键帧 caption、多段结构化时间线（chapters/events）
- 检索：帧描述+ASR+元数据一起入库，FAISS 检索 + LLM 回答
- 编辑：高光检测，支持 clips 分段导出 + reel 合集拼接
- **深度搜索**：集成 Google Custom Search API，提供网络上下文增强
- **报告生成**：整合多源分析结果，生成综合视频分析报告

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

## 4. Docker Installation
```bash
docker build -t video-agent:0.1 .
docker run --rm -p 9000:9000 -v $(pwd)/cache:/app/cache video-agent:0.1
```

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

7. Dependencies
- yt-dlp