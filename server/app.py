# server/app.py
import os
from typing import Literal, Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.skills.video_io import load_video
from agent.workflows.quick_summary import wf_quick
from agent.workflows.detailed import wf_detailed
from agent.workflows.index import wf_index
from agent.workflows.ask import wf_ask
from agent.workflows.highlights import wf_highlights
from agent.skills.persist import load_analysis

app = FastAPI(title="Video Agent Server", version="0.1.0")


# --------- Request/Response Schemas ---------

class AnalyzeReq(BaseModel):
    source_type: Literal["youtube", "url", "local"]
    uri: str

    mode: Literal["quick", "detailed"] = "detailed"
    cache_root: str = "./cache"

    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen-vl"

    max_frames: int = Field(default=128, ge=1, le=128)
    whisper_model: str = "small"


class IndexReq(BaseModel):
    source_type: Literal["youtube", "url", "local"]
    uri: str
    cache_root: str = "./cache"

    # 如果 analysis.json 不存在或缺少 asr/frames，会自动先跑 detailed（需要下面两个参数）
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen-vl"

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

    embed_base_url: str = "http://localhost:8000/v1"
    embed_model: str = "qwen-embed"


class HighlightsReq(BaseModel):
    source_type: Literal["youtube", "url", "local"]
    uri: str
    cache_root: str = "./cache"

    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen-vl"

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
                whisper_model=req.whisper_model
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
            chunk_sec=req.chunk_sec
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
            top_k=req.top_k
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
            also_make_reel=req.also_make_reel
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