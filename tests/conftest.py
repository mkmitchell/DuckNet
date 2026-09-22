"""Shared fixtures for the DuckNet test suite.

The suite is meant to run inside the Docker image (see
.claude/skills/docker-smoke), where torch, the packaged model library and the
baseline model are all present. ROOT_PATH and INSTANCE_PATH default to the
repository root so the tests can also run from a checkout that has a
models/detection/basemodel.pt.zip.
"""

import json
import os
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("ROOT_PATH", str(REPO_ROOT))
os.environ.setdefault("INSTANCE_PATH", str(REPO_ROOT))
os.environ.setdefault("CONFIG_PATH", str(REPO_ROOT / "settings.json"))
os.environ.setdefault("DO_NOT_RELOAD", "true")
# Short SSE keepalive so the stream test finishes quickly; the production
# default lives in base/backend/app.py.
os.environ.setdefault("DUCKNET_SSE_KEEPALIVE_SECONDS", "0.2")
sys.path.insert(0, str(REPO_ROOT))

MODEL_LIB = REPO_ROOT / "models_src" / "2024-10-11" / "basemodel.pt"


@pytest.fixture(scope="session")
def model_lib_path() -> pathlib.Path:
    """Directory holding the packaged model library sources."""
    if str(MODEL_LIB) not in sys.path:
        sys.path.insert(0, str(MODEL_LIB))
    return MODEL_LIB


def write_labelme(
    path: pathlib.Path, image_name: str, shapes: list[tuple[str, list[float]]]
) -> pathlib.Path:
    """Write a minimal LabelMe json of (label, [x0, y0, x1, y1]) shapes."""
    data = {
        "shapes": [
            {
                "label": label,
                "points": [[box[0], box[1]], [box[2], box[3]]],
                "shape_type": "rectangle",
            }
            for label, box in shapes
        ],
        "imagePath": image_name,
        "imageData": None,
    }
    path.write_text(json.dumps(data))
    return path
