import torch
import numpy as np
import cv2
from typing import Dict, List, Any

_processor = None
_model = None
_emotion_detector = None

def _get_audio_model():
    """Lazy-init Wav2Vec2 emotion model (local cache or HF)."""
    global _processor, _model
    if _processor is None:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
        from agent.config import get_config
        cfg = get_config()
        model_id = (cfg.get("emotion_analysis", {}).get("audio_model", None)
                    or "superb/wav2vec2-base-superb-er")
        _processor = Wav2Vec2Processor.from_pretrained(model_id)
        _model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    return _processor, _model

def _get_visual_detector():
    """Lazy-init FER detector."""
    global _emotion_detector
    if _emotion_detector is None:
        from fer import FER
        _emotion_detector = FER(mtcnn=True)
    return _emotion_detector

def analyze_audio_emotion(audio_path: str) -> Dict[str, float]:
    """
    分析音频中的情感。

    Args:
        audio_path (str): 音频文件路径。

    Returns:
        Dict: 情感概率分布。
    """
    import librosa
    processor, model = _get_audio_model()
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()

    emotions = ['neutral', 'happy', 'sad', 'angry']
    return dict(zip(emotions, probs))

def analyze_visual_emotion(frame_path: str) -> Dict[str, float]:
    """
    分析单帧中的视觉情感。

    Args:
        frame_path (str): 帧图像路径。

    Returns:
        Dict: 情感概率分布。
    """
    detector = _get_visual_detector()
    result = detector.detect_emotions(frame_path)
    if not result:
        return {}
    return result[0]['emotions']

def analyze_emotions(audio_path: str, frame_paths: List[str]) -> Dict[str, Any]:
    """
    综合分析音频和视觉情感。

    Args:
        audio_path (str): 音频路径。
        frame_paths (List[str]): 帧路径列表。

    Returns:
        Dict: 包含音频和视觉情感的分析结果。
    """
    audio_emotions = analyze_audio_emotion(audio_path)
    visual_emotions = [analyze_visual_emotion(path) for path in frame_paths]

    return {
        'audio_emotions': audio_emotions,
        'visual_emotions': visual_emotions
    }
