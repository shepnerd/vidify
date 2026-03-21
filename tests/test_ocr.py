import pytest
from agent.extensions.skills.ocr import extract_text_from_frame

def test_extract_text_from_frame():
    # Mock a frame path, assuming a test image exists
    # For real test, use a sample image
    result = extract_text_from_frame("path/to/test/frame.jpg")
    assert isinstance(result, list)