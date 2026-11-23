import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.dl.config import BATCH_SIZE, IMG_SIZE, SEED, VAL_SPLIT


def get_data(dataset_path: str):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=VAL_SPLIT,
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VAL_SPLIT,
    )

    common = dict(target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", seed=SEED)
    train_gen = train_datagen.flow_from_directory(dataset_path, subset="training", shuffle=True, **common)
    val_gen = val_datagen.flow_from_directory(dataset_path, subset="validation", shuffle=False, **common)

    labels = train_gen.classes
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(weights))

    return train_gen, val_gen, class_weights
