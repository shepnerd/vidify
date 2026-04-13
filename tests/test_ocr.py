from agent.extensions.skills.ocr import extract_text_from_frame

class _FakeOCR:
    def ocr(self, img, cls=True):
        return [[
            (
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                ("hello", 0.99),
            )
        ]]


def test_extract_text_from_frame_returns_text(monkeypatch):
    monkeypatch.setattr("agent.extensions.skills.ocr.cv2.imread", lambda _: object())
    monkeypatch.setattr("agent.extensions.skills.ocr._get_ocr", lambda: _FakeOCR())

    result = extract_text_from_frame("path/to/test/frame.jpg")

    assert result == [{
        "text": "hello",
        "bbox": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "confidence": 0.99,
    }]


def test_extract_text_from_frame_handles_missing_image(monkeypatch):
    monkeypatch.setattr("agent.extensions.skills.ocr.cv2.imread", lambda _: None)

    assert extract_text_from_frame("missing.jpg") == []
