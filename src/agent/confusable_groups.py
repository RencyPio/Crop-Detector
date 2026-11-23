CONFUSABLE_GROUPS = [
    {"Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot"},
    {"Tomato___Late_blight", "Tomato___Early_blight"},
    {"Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"},
    {"Tomato___Bacterial_spot", "Tomato___Septoria_leaf_spot", "Tomato___Early_blight", "Tomato___Leaf_Mold"},
]


def are_confusable(label_a: str, label_b: str) -> bool:
    return any(label_a in g and label_b in g for g in CONFUSABLE_GROUPS)
