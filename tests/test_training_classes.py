"""Class resolution for training: classes of interest become positives,
deselected known classes become background, unknown labels raise before
anything is mutated.
"""

import pytest


@pytest.fixture
def resolve(model_lib_path):
    import modellib

    return modellib.DuckDetector.resolve_training_classes


KNOWN = ['AMCO', 'GADW', 'Hen', 'MALL']


def test_deselected_known_classes_become_background(resolve):
    new, negative = resolve(
        KNOWN, {'GADW', 'Hen', 'Other'}, ['GADW', 'MALL'], ['Other']
    )
    assert new == []
    assert negative == ['AMCO', 'Hen', 'Other']


def test_new_class_of_interest_is_added(resolve):
    new, negative = resolve(KNOWN, {'GADW', 'RUDU'}, ['GADW', 'RUDU'], [])
    assert new == ['RUDU']
    assert negative == ['AMCO', 'Hen', 'MALL']


def test_label_outside_both_groups_raises(resolve):
    with pytest.raises(ValueError, match="BWTE"):
        resolve(KNOWN, {'GADW', 'BWTE'}, ['GADW'], [])


def test_without_classes_of_interest_every_unseen_label_is_new(resolve):
    new, negative = resolve(KNOWN, {'GADW', 'RUDU', 'Other'}, None, ['Other'])
    assert new == ['RUDU']
    assert negative == ['Other']


def test_polygon_shape_is_rejected(tmp_path, model_lib_path):
    import json

    import datasets

    path = tmp_path / 'poly.json'
    path.write_text(
        json.dumps(
            {
                'shapes': [
                    {'label': 'GADW', 'points': [[0, 0], [5, 0], [5, 5]]}
                ],
                'imagePath': 'x.jpg',
                'imageData': None,
            }
        )
    )
    with pytest.raises(ValueError, match='two-point'):
        datasets.get_boxes_from_jsonfile(str(path))
