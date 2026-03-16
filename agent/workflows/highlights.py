# agent/workflows/highlights.py
import os
from agent.skills.persist import load_analysis, save_analysis
from agent.skills.deserialize import load_transcript
from agent.skills.highlights import detect_highlights
from agent.skills.video_edit import export_highlight_clips, export_highlight_reel
from agent.workflows.detailed import wf_detailed

def wf_highlights(asset, llm_base_url: str, llm_model: str,
                  max_clips: int = 5, also_make_reel: bool = True,
                  direct_model: bool = False,
                  model_path: str = None,
                  tokenizer_path: str = None) -> dict:
    try:
        analysis = load_analysis(asset.cache_dir)
    except Exception:
        analysis = wf_detailed(asset, llm_base_url, llm_model, direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path)

    if not analysis.get("asr") or not analysis.get("timeline"):
        analysis = wf_detailed(asset, llm_base_url, llm_model, direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path)

    transcript = load_transcript(analysis["asr"])
    timeline = analysis["timeline"]

    highlights = detect_highlights(transcript, timeline, llm_model, llm_base_url, max_clips=max_clips, direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path)
    out_dir = os.path.join(asset.cache_dir, "highlights")
    highlights = export_highlight_clips(asset, highlights, out_dir)

    analysis["highlights"] = [h.model_dump() for h in highlights]

    if also_make_reel and highlights:
        reel_path = os.path.join(out_dir, "reel.mp4")
        reel_info = export_highlight_reel(highlights, reel_path)
        analysis.setdefault("artifacts", {})
        analysis["artifacts"]["reel"] = reel_info

    save_analysis(asset.cache_dir, analysis)
    return analysis