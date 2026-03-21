import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import librosa
import numpy as np
from fer import FER
import cv2
from typing import Dict, List, Any

# 初始化音频情感模型（Wav2Vec2 for emotion recognition）
processor = Wav2Vec2Processor.from_pretrained("superb/wav2vec2-base-superb-er")
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")

# 初始化视觉情感检测器
emotion_detector = FER(mtcnn=True)

def analyze_audio_emotion(audio_path: str) -> Dict[str, float]:
    """
    分析音频中的情感。

    Args:
        audio_path (str): 音频文件路径。

    Returns:
        Dict: 情感概率分布。
    """
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()

    # 假设模型输出是 ['neutral', 'happy', 'sad', 'angry'] 等
    emotions = ['neutral', 'happy', 'sad', 'angry']  # 根据模型调整
    return dict(zip(emotions, probs))

def analyze_visual_emotion(frame_path: str) -> Dict[str, float]:
    """
    分析单帧中的视觉情感。

    Args:
        frame_path (str): 帧图像路径。

    Returns:
        Dict: 情感概率分布。
    """
    result = emotion_detector.detect_emotions(frame_path)
    if not result:
        return {}

    # 返回主要情感
    emotions = result[0]['emotions']
    return emotions

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