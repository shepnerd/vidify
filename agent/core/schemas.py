# agent/schemas.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

class VideoSource(BaseModel):
    type: Literal["youtube", "url", "local"]
    uri: str

class SubtitleTrack(BaseModel):
    language: str
    source: Literal["manual", "auto"]
    format: str  # "vtt", "srt", etc.
    path: str

class ContentMetadata(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    uploader: Optional[str] = None
    upload_date: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    duration_from_source: Optional[float] = None
    view_count: Optional[int] = None
    subtitles: List[SubtitleTrack] = Field(default_factory=list)

class VideoMetadata(BaseModel):
    duration_sec: float
    fps: float
    width: int
    height: int
    has_audio: bool = True
    content: Optional[ContentMetadata] = None

class VideoAsset(BaseModel):
    id: str
    source: VideoSource
    local_path: str
    cache_dir: str
    metadata: Optional[VideoMetadata] = None
    content_metadata: Optional[ContentMetadata] = None
    subtitle_tracks: List[SubtitleTrack] = Field(default_factory=list)

class FrameItem(BaseModel):
    id: str
    ts: float
    path: str
    url: Optional[str] = None
    caption: Optional[str] = None

class FrameStrategy(BaseModel):
    type: Literal["scene", "fps", "skipped"]
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

class ContentSufficiency(BaseModel):
    asr_coverage_ratio: float
    transcript_word_count: int
    has_subtitles: bool
    has_content_metadata: bool
    is_sufficient: bool
    reason: str

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

class StreamConfig(BaseModel):
    similarity_threshold: float = 0.9
    min_segment_frames: int = 3
    max_segment_frames: int = 16
    heavy_interval: int = 5
    fps: int = 1
    source: Literal["webcam", "stream"] = "webcam"
    stream_url: Optional[str] = None

class StreamSegment(BaseModel):
    segment_id: str
    start_ts: float
    end_ts: float
    frame_paths: List[str] = Field(default_factory=list)
    caption: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StreamMemory(BaseModel):
    segments: List[StreamSegment] = Field(default_factory=list)
    global_summary: str = ""
    global_embedding: Optional[List[float]] = None
    total_frames_processed: int = 0
    total_duration_sec: float = 0.0

class AnalysisResult(BaseModel):
    video: Dict[str, Any]
    frames: Optional[FrameSet] = None
    asr: Optional[Transcript] = None
    timeline: Dict[str, Any] = Field(default_factory=dict)
    highlights: List[HighlightClip] = Field(default_factory=list)
    rag: Dict[str, Any] = Field(default_factory=dict)
    content_metadata: Optional[ContentMetadata] = None
    sufficiency: Optional[ContentSufficiency] = None
    visual_processing_skipped: bool = False
