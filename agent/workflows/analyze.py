# agent/workflows/analyze.py
from agent.workflows.brief import wf_brief
from agent.workflows.detailed import wf_detailed

def wf_analyze(asset, mode: str,
               llm_base_url: str, llm_model: str,
               max_frames: int = 128,
               whisper_model: str = "small",
               direct_model: bool = False,
               model_path: str = None,
               tokenizer_path: str = None,
               include_web_search: bool = False,
               google_api_key: str = None, google_search_engine_id: str = None) -> dict:
    if mode == "brief":
        return wf_brief(asset, llm_base_url, llm_model, max_frames=max_frames,
                        direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path,
                        include_web_search=include_web_search, google_api_key=google_api_key, 
                        google_search_engine_id=google_search_engine_id)
    if mode == "detailed":
        return wf_detailed(asset, llm_base_url, llm_model, max_frames=max_frames, whisper_model=whisper_model,
                           direct_model=direct_model, model_path=model_path, tokenizer_path=tokenizer_path,
                           include_web_search=include_web_search, google_api_key=google_api_key,
                           google_search_engine_id=google_search_engine_id)
    raise ValueError(f"Unknown analyze mode: {mode}")