"""DetectionDataset must keep boxes and labels aligned, drop rejected classes
from the targets rather than the image, and refuse labels outside the class
list instead of inventing new ids at load time.
"""

import PIL.Image
import pytest
from conftest import write_labelme


@pytest.fixture
def sample(tmp_path):
    image = tmp_path / "img.jpg"
    PIL.Image.new("RGB", (64, 48), color=(120, 120, 120)).save(image)
    annotation = write_labelme(
        tmp_path / "img.json",
        "img.jpg",
        [
            ("GADW", [1, 2, 11, 12]),
            ("Other", [20, 20, 30, 30]),
            ("Hen", [5, 5, 15, 15]),
        ],
    )
    return str(image), str(annotation)


def test_negative_class_boxes_are_removed_with_their_labels(
    sample, model_lib_path
):
    import datasets

    image, annotation = sample
    ds = datasets.DetectionDataset(
        [image],
        [annotation],
        augment=False,
        negative_classes=["Other"],
        class_list=["GADW", "Hen"],
    )
    _, target = ds[0]
    assert len(target["boxes"]) == len(target["labels"]) == 2
    assert target["labels"].tolist() == [1, 2]


def test_unknown_label_raises_instead_of_minting_an_id(sample, model_lib_path):
    import datasets

    image, annotation = sample
    ds = datasets.DetectionDataset(
        [image],
        [annotation],
        augment=False,
        negative_classes=[],
        class_list=["GADW", "Hen"],
    )
    with pytest.raises(KeyError, match="Other"):
        ds[0]
    assert ds.class_list == ["GADW", "Hen"]


def test_boxes_from_jsonfile_shape(sample, model_lib_path):
    import datasets

    _, annotation = sample
    assert datasets.get_boxes_from_jsonfile(annotation).shape == (3, 4)
    assert datasets.get_labels_from_jsonfile(annotation) == [
        "GADW",
        "Other",
        "Hen",
    ]
