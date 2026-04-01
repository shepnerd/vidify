# server/app.py
import os
import json
import queue
import threading
from typing import Literal, Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
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
from agent.extensions.workflows.live import create_live_session
from agent.core.events import event_bus, EventBus, Event

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


class LiveStartReq(BaseModel):
    source: Literal["webcam", "stream"] = "webcam"
    stream_url: Optional[str] = None
    fps: int = Field(default=1, ge=1, le=30)
    heavy_interval: int = Field(default=5, ge=1, le=30)
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen-vl"
    embed_base_url: str = "http://localhost:8000/v1"
    embed_model: str = "qwen-embed"


class LiveAskReq(BaseModel):
    session_id: str
    question: str


class LiveStopReq(BaseModel):
    session_id: str


# --------- Helpers ---------

# Live session store (in-memory)
_live_sessions: Dict[str, Any] = {}
_session_counter = 0

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


# --------- SSE Streaming Endpoint ---------

@app.post("/analyze/stream")
def analyze_stream(req: AnalyzeReq):
    """Stream analysis progress as Server-Sent Events.

    Returns SSE events for each skill start/complete/error, then a final
    'result' event with the full analysis JSON. Clients can render a live
    progress bar from the progress_pct field.
    """
    asset = _get_asset(req.source_type, req.uri, req.cache_root)
    q: queue.Queue = queue.Queue()

    def _event_to_queue(event: Event):
        q.put(event)

    def _run_analysis():
        try:
            if req.mode == "quick":
                result = wf_quick(asset, req.llm_base_url, req.llm_model, max_frames=req.max_frames)
            else:
                result = wf_detailed(
                    asset, req.llm_base_url, req.llm_model,
                    max_frames=req.max_frames,
                    whisper_model=req.whisper_model,
                    direct_model=req.direct_model,
                    model_path=req.model_path,
                    tokenizer_path=req.tokenizer_path,
                )
            q.put(("__result__", result))
        except Exception as e:
            q.put(("__error__", str(e)))

    # Subscribe to events, run analysis in background thread
    event_bus.subscribe(None, _event_to_queue)
    thread = threading.Thread(target=_run_analysis, daemon=True)
    thread.start()

    def _generate():
        while True:
            item = q.get(timeout=600)  # 10min max
            if isinstance(item, Event):
                yield item.to_sse()
            elif isinstance(item, tuple):
                tag, payload = item
                if tag == "__result__":
                    yield f"event: result\ndata: {json.dumps(payload, default=str)}\n\n"
                    break
                elif tag == "__error__":
                    yield f"event: error\ndata: {json.dumps({'error': payload})}\n\n"
                    break
        event_bus.unsubscribe(None, _event_to_queue)

    return StreamingResponse(_generate(), media_type="text/event-stream")


# --------- Live Streaming Endpoints ---------

@app.post("/live/start")
def live_start(req: LiveStartReq):
    """Start a live stream processing session."""
    global _session_counter
    _session_counter += 1
    session_id = f"live_{_session_counter:04d}"

    cfg = {
        "source": req.source,
        "stream_url": req.stream_url,
        "fps": req.fps,
        "heavy_interval": req.heavy_interval,
        "llm_base_url": req.llm_base_url,
        "llm_model": req.llm_model,
        "embed_base_url": req.embed_base_url,
        "embed_model": req.embed_model,
    }
    try:
        session = create_live_session(session_id, cfg)
        session.start()
        _live_sessions[session_id] = session
        return {"session_id": session_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/live/ask")
def live_ask(req: LiveAskReq):
    """Ask a question about the current live stream."""
    session = _live_sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found")
    try:
        return session.ask(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/live/stop")
def live_stop(req: LiveStopReq):
    """Stop a live stream session and return final memory."""
    session = _live_sessions.pop(req.session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found")
    try:
        return session.stop()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/live/status/{session_id}")
def live_status(session_id: str):
    """Get current status of a live stream session."""
    session = _live_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session.status()