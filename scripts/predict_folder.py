import csv
import sys
from pathlib import Path

from src.dl.predict import Predictor

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main(folder: Path, out_csv: Path):
    predictor = Predictor()
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)
    print(f"Found {len(images)} images in {folder}\n")

    rows = []
    for p in images:
        top3 = predictor.top_k(predictor.predict(p), 3)
        print(p.name)
        for rank, t in enumerate(top3, 1):
            print(f"   {rank}. {t['label']}  ({t['prob'] * 100:.1f}%)")
        rows.append([p.name] + [v for t in top3 for v in (t["label"], t["prob"])])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "top1", "p1", "top2", "p2", "top3", "p3"])
        w.writerows(rows)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/real_world_test_images")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "reports/predictions.csv")
    main(folder, out)
