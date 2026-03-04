# agent/skills/deserialize.py
from agent.schemas import FrameSet, Transcript

def load_frames(frames_dict: dict) -> FrameSet:
    return FrameSet.model_validate(frames_dict)

def load_transcript(asr_dict: dict) -> Transcript:
    return Transcript.model_validate(asr_dict)