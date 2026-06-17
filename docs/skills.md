# Skills Reference

Skills are self-contained processing units in `agent/extensions/skills/`. Each skill does one thing and exposes simple function-level APIs.

## Video I/O & Processing

### video_io — Video Loading
```python
load_video(source_type: str, uri: str, cache_root: str) -> VideoAsset
```
Handles `youtube`, `url`, and `local` sources. Downloads and caches the video, returns a `VideoAsset` with `id`, `source`, `local_path`, and `cache_dir`.

### video_download — YouTube / URL Download
```python
download_youtube(uri: str, out_dir: str) -> str   # Returns local path
download_generic(uri: str, out_dir: str) -> str
```
Uses `yt-dlp` to download videos. Caches as `source.mp4` — skips download if file already exists.

### video_probe — Metadata Extraction
```python
probe_video(video: VideoAsset) -> VideoMetadata
```
Runs `ffprobe` to extract duration, fps, resolution, and audio presence.

### audio_extract — Audio Extraction
```python
extract_audio(video: VideoAsset) -> str   # Returns path to audio.wav
```
Extracts audio to 16kHz mono WAV using FFmpeg. Reuses cached file.

### frame_sampler — Frame Sampling
```python
sample_frames(video, out_dir: str, strategy: FrameStrategy) -> FrameSet
```
Two strategies:
- **fps** — sample at fixed rate (e.g., 1 fps)
- **scene** — scene-change detection (configurable threshold)

Frames are resized to 256x144. Returns a `FrameSet` with timestamps.

### video_edit — Clip Export
```python
export_highlight_clips(video, highlights, out_dir) -> list
export_highlight_reel(clips, out_path) -> dict
```
Exports individual highlight clips and optionally concatenates them into a reel using FFmpeg concat demuxer.

## Visual Understanding

### vision_caption — Frame / Video Captioning
```python
caption_frame(frame_path, model_name, base_url) -> str
caption_frames(frames, model_name, base_url, ...) -> FrameSet
caption_video(video_path, model_name, base_url, max_duration=60) -> list
caption_video_as_frameset(video_path, ...) -> FrameSet
```
- `caption_frame()` — convenience wrapper for one frame, used by live-stream processing
- `caption_frames()` — batched frame captioning via multimodal model
- `caption_video()` — direct video input captioning (Qwen3-VL native video support), auto-splits long videos into segments
- Supports both vLLM and direct model loading

### object_detection — Object Detection
```python
detect_objects_in_video_frames(frames, model="yolov8n.pt", conf=0.5) -> list
```
YOLOv8 nano model. Returns class labels, confidence scores, and bounding boxes per frame.

### ocr — Text Extraction
```python
extract_text_from_video_frames(frames, lang="ch") -> list
```
PaddleOCR with multi-language support. Returns text, position, and confidence per frame.

### emotion_analysis — Emotion Detection
```python
analyze_emotions(video, audio_path, frames) -> dict
```
Combined audio (Wav2Vec2) + visual (FER) emotion analysis. Returns probability distributions over emotion categories.

## Audio Processing

### asr — Speech Recognition
```python
transcribe(audio_path, model_size="small") -> Transcript
```
Uses HuggingFace Transformers Whisper by default, with a local faster-whisper directory fallback when available. Returns `Transcript` with segment-level timing, language detection, and confidence scores.

### translation — Multi-language Translation
```python
translate_asr_results(transcript, source_lang="en", target_lang="zh") -> list
```
Uses Helsinki-NLP OPUS models. Translates ASR segments while preserving timing.

## Analysis & Understanding

### timeline_builder — Timeline Generation
```python
build_timeline(metadata, transcript, frames, model_name, base_url) -> dict
```
Uses LLM to create structured chapters (with titles and summaries) and events (with evidence references) from video content.

### highlights — Highlight Detection
```python
detect_highlights(analysis, model_name, base_url) -> list[HighlightClip]
```
LLM-based identification of high-information-density segments. Detects key conclusions, turning points, and visually striking moments.

### mm_qa — Multimodal Q&A
```python
mm_qa(frames, question, model_name, base_url) -> str
```
Direct multimodal question-answering using frame images + text prompt.

## Retrieval & Indexing

### rag_index — Time-based Chunking
```python
chunk_by_time(analysis, chunk_sec=20) -> list
```
Splits video content into time-based chunks, organizing frames and ASR segments by chunk.

### rag_faiss — FAISS Semantic Index
```python
build_faiss_index(chunks, embed_base_url, embed_model) -> dict
search_faiss(query, index_dir, embed_base_url, embed_model, top_k=5) -> list
```
Creates dense embeddings via OpenAI-compatible API, builds FAISS index with L2 normalization. Semantic search returns chunks with metadata (frame IDs, segment IDs, timing).

## Web & Information

### web_search — Web Search Integration
```python
web_search(query) -> list
deep_search_enhance(analysis, google_api_key, google_search_engine_id) -> list
google_search(query, api_key, engine_id) -> list
baidu_search(query) -> list
```
Google Custom Search with automatic Baidu fallback for Chinese users. Includes network detection for regional optimization.

## Specialized Processing

### batch_processing — Parallel Video Processing
```python
process_video_batch(video_paths, mode, config, max_workers=4) -> list
```
ThreadPoolExecutor-based parallel processing with error isolation per video.

### live_stream_processing — Real-time Stream Processing
```python
process_live_stream(source, callback, resolution, fps, heavy_interval)
```
Supports webcam and stream URLs. SlowFast strategy alternates heavy and light models for efficiency.

### custom_summary — Customizable Summarization
```python
custom_summary(analysis, style, preferences) -> str
```
Multiple styles: educational, entertainment, technical. Supports user preference handling for length and focus control.

## Persistence

### persist — Cache Management
```python
save_analysis(cache_dir, result) -> None
load_analysis(cache_dir) -> dict
```
Saves/loads `analysis.json` in the video's cache directory.

### deserialize — Object Reconstruction
```python
load_frames(data) -> FrameSet
load_transcript(data) -> Transcript
```
Pydantic-based reconstruction of data models from JSON.
