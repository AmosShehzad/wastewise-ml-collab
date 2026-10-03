from wastewise_ml_collab.data_utils import count_images_by_class


def test_count_images_by_class(tmp_path):
    class_a = tmp_path / "cardboard"
    class_b = tmp_path / "plastic"

    class_a.mkdir()
    class_b.mkdir()

    (class_a / "image1.jpg").touch()
    (class_a / "image2.jpg").touch()
    (class_b / "image1.jpg").touch()

    result = count_images_by_class(tmp_path)

    assert result["cardboard"] == 2
    assert result["plastic"] == 1
