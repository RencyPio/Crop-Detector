import sys
from pathlib import Path

from src.dl.predict import Predictor

EXTS = {".jpg", ".jpeg", ".png"}


def main(root: Path):
    predictor = Predictor()
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    unknown = [p.name for p in folders if p.name not in predictor.class_names]
    print(f"Class folders: {len(folders)} (not a known class: {unknown})\n")

    correct = total = 0
    wrong = []
    for class_dir in folders:
        for img_path in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in EXTS):
            top1 = predictor.top_k(predictor.predict(img_path), 1)[0]
            total += 1
            if top1["label"] == class_dir.name:
                correct += 1
            else:
                wrong.append((class_dir.name, top1["label"], top1["prob"]))

    print(f"Correct: {correct}/{total} = {correct / max(total, 1) * 100:.1f}%\n")
    for true, pred, conf in wrong:
        print(f"WRONG  true={true}\n       pred={pred} ({conf * 100:.1f}%)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/sanity_check"))
