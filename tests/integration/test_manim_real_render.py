"""Optional: renders with a real Manim install. Skipped unless MANIM_PYTHON points at an interpreter that has manim + ffmpeg."""
import os
import shutil
from pathlib import Path

import pytest

from app.visualization.manim_templates import validate_spec
from app.visualization.policy import FailureCategory
from app.visualization.renderer import is_valid_mp4, render_spec

PY = os.environ.get("MANIM_PYTHON", "")
pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not PY or not Path(PY).exists(), reason="MANIM_PYTHON not set")]
SPEC = {"template": "sorting", "title": "Bubble", "params": {"algorithm": "bubble", "array": [3, 1, 2]}}


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_real_render_produces_valid_mp4():
    res = await render_spec(validate_spec(SPEC), python=PY, timeout_seconds=170)
    try:
        assert res.ok, res.detail
        assert is_valid_mp4(res.path, 20 * 1024 * 1024)
    finally:
        if res.path:
            res.path.unlink(missing_ok=True)


async def test_real_render_timeout_kills_process():
    res = await render_spec(validate_spec(SPEC), python=PY, timeout_seconds=1)
    assert not res.ok and res.category == FailureCategory.TIMEOUT
