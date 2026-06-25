import pytest
from fastapi import HTTPException


def test_safe_upload_name_strips_path_components():
    from server.app import _safe_upload_name

    name = _safe_upload_name("../../my video.mp4")

    assert "/" not in name
    assert "\\" not in name
    assert name.startswith("my_video_")
    assert name.endswith(".mp4")


def test_safe_upload_name_rejects_unsupported_extension():
    from server.app import _safe_upload_name

    with pytest.raises(HTTPException) as exc:
        _safe_upload_name("notes.txt")

    assert exc.value.status_code == 400
