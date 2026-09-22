"""CSV output of the command line interface must match the README contract:

semicolon separated, Class holds the four-letter species code, confidence on a
0 to 1 scale, and every row has the same number of fields as the header.
"""

import numpy as np
import pytest

import backend.cli as cli
import backend.settings as settings


@pytest.fixture
def fake_results(tmp_path):
    image = tmp_path / "IMG_0001.jpg"
    image.write_bytes(b"not really a jpeg")
    per_class = [{"GADW": 0.91, "MALL": 0.02}, {"Hen": 0.55, "GADW": 0.1}]
    return [
        {
            "filename": str(image),
            "result": {
                "labels": ["GADW", "Hen"],
                "per_class_scores": per_class,
                "boxes": np.array(
                    [[10.0, 20.0, 30.0, 40.0], [1.5, 2.5, 3.5, 4.5]]
                ),
            },
        }
    ]


def rows(csv_text):
    """Split CSV lines into fields, dropping the one trailing separator."""
    return [
        line.removesuffix(";").split(";")
        for line in csv_text.strip().splitlines()
    ]


def test_species_code_column_uses_model_label_codes(fake_results):
    parsed = rows(cli.results_to_csv(fake_results))
    header, first, second = parsed
    assert header == ["Filename", "Date", "Time", "Class", "Confidence level"]
    assert first[3] == "GADW"
    assert second[3] == "Hen"


def test_confidence_is_on_zero_to_one_scale(fake_results):
    parsed = rows(cli.results_to_csv(fake_results))
    assert parsed[1][4] == "0.91"
    assert parsed[2][4] == "0.55"


def test_boxes_column_and_row_lengths(fake_results):
    parsed = rows(cli.results_to_csv(fake_results, export_boxes=True))
    assert parsed[0][-1] == "Box"
    assert all(len(row) == len(parsed[0]) for row in parsed)
    assert parsed[1][5] == "10.0 20.0 30.0 40.0"


def test_empty_result_still_emits_one_row(fake_results):
    fake_results[0]["result"] = {
        "labels": [],
        "per_class_scores": [],
        "boxes": np.zeros((0, 4)),
    }
    parsed = rows(cli.results_to_csv(fake_results, export_boxes=True))
    assert len(parsed) == 2
    assert len(parsed[1]) == len(parsed[0])


def test_scientific_name_labels_are_mapped_to_codes(fake_results):
    codes = settings.parse_species_codes_file()
    assert codes["Mareca strepera"] == "GADW"
    assert cli.species_code_for("Mareca strepera", codes) == "GADW"
    assert cli.species_code_for("GADW", codes) == "GADW"
    assert cli.species_code_for("Something new", codes) == "Something new"
