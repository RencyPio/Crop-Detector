from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.dl.config import IMG_SIZE, NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, weights: str | None = "imagenet"):
    base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights=weights)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model, base_model


def unfreeze_for_finetuning(model, base_model, n_layers: int = 30, lr: float = 1e-5):
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    for layer in base_model.layers[-n_layers:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model
