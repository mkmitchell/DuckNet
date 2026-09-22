"""The train/validation split must depend only on random_state, not on the
per-process string hash seed, and an empty split must fail loudly.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest
from conftest import MODEL_LIB, REPO_ROOT, write_labelme

SPLIT_SCRIPT = pathlib.Path(__file__).with_name("split_once.py")


def make_dataset(tmp_path, per_class=25):
    images, jsons = [], []
    for cls in ["GADW", "MALL", "Hen"]:
        for i in range(per_class):
            image = tmp_path / f"{cls}_{i}.jpg"
            image.write_bytes(b"")
            annotation = write_labelme(
                tmp_path / f"{cls}_{i}.json",
                image.name,
                [(cls, [0, 0, 10, 10])],
            )
            images.append(str(image))
            jsons.append(str(annotation))
    return images, jsons


def run_split_in_subprocess(images, jsons, hash_seed):
    env = dict(
        os.environ, PYTHONHASHSEED=str(hash_seed), PYTHONPATH=str(MODEL_LIB)
    )
    payload = json.dumps({"images": images, "jsons": jsons})
    result = subprocess.run(
        [sys.executable, str(SPLIT_SCRIPT)],
        input=payload,
        text=True,
        capture_output=True,
        env=env,
        cwd=REPO_ROOT,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_split_is_identical_across_hash_seeds(tmp_path):
    images, jsons = make_dataset(tmp_path)
    first = run_split_in_subprocess(images, jsons, hash_seed=1)
    second = run_split_in_subprocess(images, jsons, hash_seed=2)
    assert first == second
    assert len(first["train"]) > 0 and len(first["test"]) > 0
    assert set(first["train"]).isdisjoint(first["test"])


def test_split_with_no_eligible_class_raises(tmp_path, model_lib_path):
    import modellib

    images, jsons = make_dataset(tmp_path, per_class=3)
    detector = modellib.DuckDetector.__new__(modellib.DuckDetector)
    with pytest.raises(ValueError, match="fewer than"):
        detector._stratified_split(images, jsons)
