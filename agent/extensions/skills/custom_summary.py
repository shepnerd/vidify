from typing import Dict, Any, List
from agent.skills.mm_qa import ask_llm  # 假设有LLM问答函数

def generate_custom_summary(video_analysis: Dict[str, Any], style: str = 'general', preferences: Dict[str, Any] = None) -> str:
    """
    生成自定义风格的视频摘要。

    Args:
        video_analysis (Dict): 视频分析结果。
        style (str): 摘要风格，如'educational', 'entertainment', 'technical'。
        preferences (Dict): 用户偏好，如长度、重点等。

    Returns:
        str: 生成的摘要。
    """
    if preferences is None:
        preferences = {}

    # 构建提示
    prompt = f"根据以下视频分析结果，生成一个{style}风格的摘要。"
    if 'length' in preferences:
        prompt += f" 摘要长度约为{preferences['length']}字。"
    if 'focus' in preferences:
        prompt += f" 重点关注{preferences['focus']}。"

    prompt += f"\n\n分析结果：{video_analysis}"

    # 使用LLM生成摘要
    summary = ask_llm(prompt, context=video_analysis)
    return summary

def generate_multiple_summaries(video_analysis: Dict[str, Any], styles: List[str]) -> Dict[str, str]:
    """
    生成多种风格的摘要。

    Args:
        video_analysis (Dict): 视频分析结果。
        styles (List[str]): 风格列表。

    Returns:
        Dict: 风格到摘要的映射。
    """
    summaries = {}
    for style in styles:
        summaries[style] = generate_custom_summary(video_analysis, style)
    return summaries