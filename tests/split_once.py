"""Helper for test_split.py: run one stratified split in a fresh interpreter.

Reads {"images": [...], "jsons": [...]} from stdin and prints the split as a
single JSON line so the test can compare runs made under different
PYTHONHASHSEED values.
"""

import json
import sys

import modellib


def main() -> None:
    payload = json.load(sys.stdin)
    detector = modellib.DuckDetector.__new__(modellib.DuckDetector)
    train, test, _, _ = detector._stratified_split(
        payload["images"], payload["jsons"]
    )
    print(json.dumps({"train": train, "test": test}))


if __name__ == "__main__":
    main()
