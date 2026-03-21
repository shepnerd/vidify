# server/app.py
import os
from typing import Literal, Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from agent.extensions.skills.video_io import load_video
from agent.extensions.workflows.quick_summary import wf_quick
from agent.extensions.workflows.detailed import wf_detailed
from agent.extensions.workflows.index import wf_index
from agent.extensions.workflows.ask import wf_ask
from agent.extensions.workflows.highlights import wf_highlights
from agent.extensions.skills.persist import load_analysis

app = FastAPI(title="Video Agent Server", version="0.1.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create directories if not exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)


# --------- Request/Response Schemas ---------

class AnalyzeReq(BaseModel):
    source_type: Literal["youtube", "url", "local"]
    uri: str

    mode: Literal["quick", "detailed"] = "detailed"
    cache_root: str = "./cache"

    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen-vl"

    direct_model: bool = False
    model_path: str = "/models/qwen-vl"
    tokenizer_path: Optional[str] = None

    max_frames: int = Field(default=128, ge=1, le=128)
    whisper_model: str = "small"


class IndexReq(BaseModel):
    source_type: Literal["youtube", "url", "local"]
    uri: str
    cache_root: str = "./cache"

    # 如果 analysis.json 不存在或缺少 asr/frames，会自动先跑 detailed（需要下面两个参数）
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen-vl"

    direct_model: bool = False
    model_path: str = "/models/qwen-vl"
    tokenizer_path: Optional[str] = None

    # embeddings（vLLM OpenAI-compatible 支持 /v1/embeddings）[1]
    embed_base_url: str = "http://localhost:8000/v1"
    embed_model: str = "qwen-embed"

    chunk_sec: int = Field(default=20, ge=5, le=120)


class AskReq(BaseModel):
    source_type: Literal["youtube", "url", "local"]
    uri: str
    cache_root: str = "./cache"

    question: str
    top_k: int = Field(default=5, ge=1, le=20)

    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen-vl"

    direct_model: bool = False
    model_path: str = "/models/qwen-vl"
    tokenizer_path: Optional[str] = None

    embed_base_url: str = "http://localhost:8000/v1"
    embed_model: str = "qwen-embed"


class HighlightsReq(BaseModel):
    source_type: Literal["youtube", "url", "local"]
    uri: str
    cache_root: str = "./cache"

    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen-vl"

    direct_model: bool = False
    model_path: str = "/models/qwen-vl"
    tokenizer_path: Optional[str] = None

    max_clips: int = Field(default=5, ge=1, le=20)
    also_make_reel: bool = True


class LoadAnalysisReq(BaseModel):
    source_type: Literal["youtube", "url", "local"]
    uri: str
    cache_root: str = "./cache"


# --------- Helpers ---------

def _get_asset(source_type: str, uri: str, cache_root: str):
    try:
        return load_video(source_type, uri, cache_root)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _as_http_error(e: Exception):
    return HTTPException(status_code=500, detail=str(e))


# --------- Endpoints ---------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
def analyze(req: AnalyzeReq):
    """
    quick: scene关键帧 + caption + timeline（无ASR）
    detailed: scene关键帧 + caption + ASR + timeline

    注意：caption 使用 image_url 传本地帧路径；vLLM 需要 --allowed-local-media-path 放行缓存目录 [1]
    """
    asset = _get_asset(req.source_type, req.uri, req.cache_root)

    try:
        if req.mode == "quick":
            out = wf_quick(asset, req.llm_base_url, req.llm_model, max_frames=req.max_frames)
        else:
            out = wf_detailed(
                asset, req.llm_base_url, req.llm_model,
                max_frames=req.max_frames,
                whisper_model=req.whisper_model,
                direct_model=req.direct_model,
                model_path=req.model_path,
                tokenizer_path=req.tokenizer_path
            )
        return out
    except Exception as e:
        raise _as_http_error(e)

@app.post("/index")
def index(req: IndexReq):
    """
    基于 analysis.json 构建 FAISS 索引（帧描述+ASR+元数据）。
    若 analysis 不存在或不完整，会自动补跑 detailed（使用 llm_base_url/llm_model）再建索引。
    embeddings 使用 OpenAI-compatible /v1/embeddings [1]
    """
    asset = _get_asset(req.source_type, req.uri, req.cache_root)
    try:
        rag = wf_index(
            asset,
            llm_base_url=req.llm_base_url, llm_model=req.llm_model,
            embed_base_url=req.embed_base_url, embed_model=req.embed_model,
            chunk_sec=req.chunk_sec,
            direct_model=req.direct_model,
            model_path=req.model_path,
            tokenizer_path=req.tokenizer_path
        )
        return {"rag": rag, "cache_dir": asset.cache_dir}
    except Exception as e:
        raise _as_http_error(e)

@app.post("/ask")
def ask(req: AskReq):
    """
    需要先 /index（或你让 /index 自动补跑 detailed）。
    返回：answer + evidence（带时间区间、frame_ids、asr_segment_ids）+ hits
    """
    asset = _get_asset(req.source_type, req.uri, req.cache_root)
    try:
        out = wf_ask(
            asset, req.question,
            llm_base_url=req.llm_base_url, llm_model=req.llm_model,
            embed_base_url=req.embed_base_url, embed_model=req.embed_model,
            top_k=req.top_k,
            direct_model=req.direct_model,
            model_path=req.model_path,
            tokenizer_path=req.tokenizer_path
        )
        return out
    except Exception as e:
        raise _as_http_error(e)

@app.post("/highlights")
def highlights(req: HighlightsReq):
    """
    生成高光 clips，并可选拼接 reel。
    若 analysis 不存在或缺少 asr/timeline，会自动补跑 detailed。
    """
    asset = _get_asset(req.source_type, req.uri, req.cache_root)
    try:
        out = wf_highlights(
            asset,
            llm_base_url=req.llm_base_url, llm_model=req.llm_model,
            max_clips=req.max_clips,
            also_make_reel=req.also_make_reel,
            direct_model=req.direct_model,
            model_path=req.model_path,
            tokenizer_path=req.tokenizer_path
        )
        return out
    except Exception as e:
        raise _as_http_error(e)

@app.post("/analysis")
def get_analysis(req: LoadAnalysisReq):
    """
    读取已落盘的 analysis.json（用于前端可视化时间线）。
    """
    asset = _get_asset(req.source_type, req.uri, req.cache_root)
    try:
        return load_analysis(asset.cache_dir)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"analysis.json not found or unreadable: {e}")

# GUI Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(request: Request, file: UploadFile = File(...), mode: str = Form(...)):
    # Save uploaded file
    file_path = f"cache/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Process video
    asset = load_video("local", file_path, "cache")
    result = run(asset, mode, {"llm_base_url": "http://localhost:8000/v1", "llm_model": "qwen-vl"})

    return templates.TemplateResponse("result.html", {"request": request, "result": result})