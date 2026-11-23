import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.dl.config import KAGGLE_DATASET_PATH, KAGGLE_OUTPUT_DIR
from src.dl.model import build_model, unfreeze_for_finetuning
from src.dl.preprocessing import get_data


def plot_history(histories, out_path):
    acc = sum((h.history["accuracy"] for h in histories), [])
    val_acc = sum((h.history["val_accuracy"] for h in histories), [])
    loss = sum((h.history["loss"] for h in histories), [])
    val_loss = sum((h.history["val_loss"] for h in histories), [])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train"); plt.plot(val_acc, label="Val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train"); plt.plot(val_loss, label="Val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


def evaluate(model, val_gen, out_dir):
    val_gen.reset()
    y_prob = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = val_gen.classes
    names = list(val_gen.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=names, digits=3)
    print(report)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), confusion_matrix(y_true, y_pred))


def main(dataset_path=KAGGLE_DATASET_PATH, out_dir=KAGGLE_OUTPUT_DIR, head_epochs=20, ft_epochs=8):
    train_gen, val_gen, class_weights = get_data(dataset_path)
    model, base_model = build_model(num_classes=train_gen.num_classes)

    h1 = model.fit(
        train_gen, validation_data=val_gen, epochs=head_epochs, class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            ModelCheckpoint(os.path.join(out_dir, "crop_disease_cnn_best.keras"),
                            monitor="val_loss", save_best_only=True, verbose=1),
        ],
    )

    unfreeze_for_finetuning(model, base_model, n_layers=30, lr=1e-5)
    h2 = model.fit(
        train_gen, validation_data=val_gen, epochs=ft_epochs, class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ModelCheckpoint(os.path.join(out_dir, "crop_disease_cnn_ft_best.keras"),
                            monitor="val_loss", save_best_only=True, verbose=1),
        ],
    )

    model.save(os.path.join(out_dir, "crop_disease_cnn_ft_final.keras"))
    with open(os.path.join(out_dir, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f)

    plot_history([h1, h2], os.path.join(out_dir, "training_curves.png"))
    evaluate(model, val_gen, out_dir)


if __name__ == "__main__":
    main()
