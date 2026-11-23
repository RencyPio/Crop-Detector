from src.dl.labels import (crop_of, is_healthy, list_crops, load_class_names, pretty_label,
                           split_label)


def test_38_classes_in_index_order():
    names = load_class_names()
    assert len(names) == 38
    assert names[0] == "Apple___Apple_scab"
    assert names[30] == "Tomato___Late_blight"
    assert names[37] == "Tomato___healthy"


def test_split_and_pretty():
    assert split_label("Corn_(maize)___Common_rust_") == ("Corn_(maize)", "Common_rust_")
    assert pretty_label("Corn_(maize)___Common_rust_") == "Corn (maize) - Common rust"
    assert crop_of("Pepper,_bell___Bacterial_spot") == "Pepper,_bell"


def test_healthy_count():
    names = load_class_names()
    assert sum(is_healthy(n) for n in names) == 12
    assert not is_healthy("Tomato___Late_blight")


def test_14_crops():
    crops = list_crops()
    assert len(crops) == 14
    assert "Tomato" in crops and "Corn_(maize)" in crops
