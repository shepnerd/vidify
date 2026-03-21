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