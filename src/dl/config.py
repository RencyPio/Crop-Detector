from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "crop_disease_cnn_ft_final.keras"
TFLITE_PATH = MODELS_DIR / "crop_disease_cnn.tflite"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.15
SEED = 42
NUM_CLASSES = 38

KAGGLE_DATASET_PATH = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"
KAGGLE_OUTPUT_DIR = "/kaggle/working"
