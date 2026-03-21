from transformers import pipeline
from typing import List, Dict, Any

# 初始化翻译管道（支持多语言，如英译中）
translator_en_zh = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
translator_zh_en = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")

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
    if source_lang == 'en' and target_lang == 'zh':
        result = translator_en_zh(text)
    elif source_lang == 'zh' and target_lang == 'en':
        result = translator_zh_en(text)
    else:
        # 可扩展更多语言对
        raise ValueError(f"Unsupported language pair: {source_lang} to {target_lang}")

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
        translated_text = translate_text(segment['text'], target_lang=target_lang)
        new_segment = segment.copy()
        new_segment['translated_text'] = translated_text
        translated.append(new_segment)
    return translated