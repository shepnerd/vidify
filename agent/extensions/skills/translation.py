from typing import List, Dict, Any

_translators = {}

def _get_translator(source_lang: str, target_lang: str):
    """Lazy-init translation pipeline for a language pair (HF-managed)."""
    key = f"{source_lang}-{target_lang}"
    if key not in _translators:
        from transformers import pipeline
        model_map = {
            "en-zh": "Helsinki-NLP/opus-mt-en-zh",
            "zh-en": "Helsinki-NLP/opus-mt-zh-en",
        }
        if key not in model_map:
            raise ValueError(f"Unsupported language pair: {source_lang} to {target_lang}")
        _translators[key] = pipeline("translation", model=model_map[key])
    return _translators[key]

def translate_text(text: str, source_lang: str = 'en', target_lang: str = 'zh') -> str:
    """
    翻译文本。

    Args:
        text (str): 待翻译文本。
        source_lang (str): 源语言。
        target_lang (str): 目标语言。

    Returns:
        str: 翻译后的文本。
    """
    translator = _get_translator(source_lang, target_lang)
    result = translator(text)
    return result[0]['translation_text']

def translate_asr_results(asr_results: List[Dict[str, Any]], target_lang: str = 'zh') -> List[Dict[str, Any]]:
    """
    翻译ASR结果。

    Args:
        asr_results (List[Dict]): ASR结果列表，每个包含'text'等。
        target_lang (str): 目标语言。

    Returns:
        List[Dict]: 翻译后的ASR结果。
    """
    translated = []
    for segment in asr_results:
        # Support both dict and Pydantic ASRSegment objects
        text = segment['text'] if isinstance(segment, dict) else segment.text
        translated_text = translate_text(text, target_lang=target_lang)
        if isinstance(segment, dict):
            new_segment = segment.copy()
        else:
            new_segment = segment.model_dump() if hasattr(segment, 'model_dump') else dict(segment)
        new_segment['translated_text'] = translated_text
        translated.append(new_segment)
    return translated
