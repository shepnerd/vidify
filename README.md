# VidCopilot

一个视频处理/分析/理解 Agent：
- 输入：YouTube URL（优先），后续可扩展任意 URL、本地视频
- 处理：下载缓存、ffmpeg 解码、场景切分抽关键帧（默认<=128）、音频提取、ASR
- 理解：关键帧 caption、多段结构化时间线（chapters/events）
- 检索：帧描述+ASR+元数据一起入库，FAISS 检索 + LLM 回答
- 编辑：高光检测，支持 clips 分段导出 + reel 合集拼接

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

## 3. Docker Installation
```bash
docker build -t video-agent:0.1 .
docker run --rm -p 9000:9000 -v $(pwd)/cache:/app/cache video-agent:0.1
```

## 4. API用法
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

5. 端到端 Demo 脚本

6. Dependencies
- yt-dlp