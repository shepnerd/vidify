# agent/workflows/report.py
import os
from typing import Dict, Any, List
from agent.extensions.skills.persist import load_analysis
from agent.extensions.skills.web_search import deep_search_enhance
from agent.core.schemas import VideoAsset

def generate_report(asset: VideoAsset, analysis_type: str = "brief", 
                   include_web_search: bool = True, 
                   llm_base_url: str = None, llm_model: str = None,
                   direct_model: bool = False, model_path: str = None, tokenizer_path: str = None,
                   google_api_key: str = None, google_search_engine_id: str = None) -> Dict[str, Any]:
    """
    Generate a comprehensive report based on video analysis.
    
    Args:
        asset: The video asset
        analysis_type: Type of analysis to base report on ("brief" or "detailed")
        include_web_search: Whether to include web search enhancement
        llm_base_url: LLM base URL
        llm_model: LLM model name
        direct_model: Whether to use direct model
        model_path: Path to model
        tokenizer_path: Path to tokenizer
        
    Returns:
        Dictionary containing the comprehensive report
    """
    
    # Load existing analysis
    analysis_file = os.path.join(asset.cache_dir, "analysis.json")
    if os.path.exists(analysis_file):
        analysis = load_analysis(asset.cache_dir)
    else:
        # If no analysis exists, perform brief analysis first
        from agent.extensions.workflows.brief import wf_brief
        analysis = wf_brief(asset, llm_base_url, llm_model, direct_model=direct_model, 
                           model_path=model_path, tokenizer_path=tokenizer_path)
    
    # Extract key information
    video_info = analysis.get("video", {})
    timeline = analysis.get("timeline", "")
    frame_items = _get_frame_items(analysis.get("frames"))
    asr = analysis.get("asr", {})
    
    # Prepare search query based on video content
    search_query = f"video analysis {video_info.get('title', 'unknown video')} {timeline[:200]}"
    
    # Perform deep search if requested
    web_search_results = {}
    if include_web_search:
        try:
            web_search_results = deep_search_enhance(search_query, timeline[:500],
                                                   api_key=google_api_key,
                                                   search_engine_id=google_search_engine_id)
        except Exception as e:
            web_search_results = {"error": str(e)}
    
    # Generate comprehensive report
    report = {
        "video_metadata": video_info,
        "summary": {
            "duration": video_info.get("duration", 0),
            "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
            "frame_count": len(frame_items),
            "has_transcript": bool(asr and asr.get("segments")),
            "language": asr.get("language") if asr else None
        },
        "timeline_summary": timeline,
        "key_frames": [
            {
                "timestamp": frame.get("ts", 0),
                "description": frame.get("caption", "")
            } for frame in frame_items[:10]
        ],
        "transcript_highlights": extract_transcript_highlights(asr) if asr else [],
        "web_search_insights": web_search_results,
        "recommendations": generate_recommendations(analysis, web_search_results),
        "analysis_type": analysis_type
    }
    
    # Save report
    report_file = os.path.join(asset.cache_dir, "report.json")
    import json
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report

def _get_frame_items(frames: Any) -> List[Dict[str, Any]]:
    """Normalize stored FrameSet payloads into a flat list of frame dicts."""
    if isinstance(frames, dict):
        items = frames.get("items", [])
        return items if isinstance(items, list) else []
    if isinstance(frames, list):
        return frames
    return []

def extract_transcript_highlights(asr: Dict) -> List[Dict]:
    """Extract key highlights from transcript."""
    segments = asr.get("segments", [])
    highlights = []
    
    # Simple extraction: take segments with significant text
    for segment in segments:
        text = segment.get("text", "").strip()
        if len(text) > 20:  # Significant text
            highlights.append({
                "timestamp": segment.get("start", 0),
                "text": text
            })
    
    return highlights[:10]  # Limit to top 10

def generate_recommendations(analysis: Dict, web_search: Dict) -> List[str]:
    """Generate recommendations based on analysis and web search."""
    recommendations = []
    frame_count = len(_get_frame_items(analysis.get("frames")))
    
    # Basic recommendations based on analysis
    if not analysis.get("asr"):
        recommendations.append("Consider adding audio transcription for better understanding")
    
    if frame_count < 50:
        recommendations.append("Increase frame sampling for more detailed visual analysis")
    
    # Add web search based recommendations
    if web_search and "search_results" in web_search:
        recommendations.append("Web search results suggest additional context available online")
    
    return recommendations
