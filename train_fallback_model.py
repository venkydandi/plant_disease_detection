from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ml_utils import IMG_SIZE, extract_features, normalize_image


@dataclass
class TrainConfig:
    dataset_dir: Path = Path(r"C:\Users\HP\plant_dataset")
    output_path: Path = Path("fallback_model.joblib")
    test_size: float = 0.2
    random_state: int = 42
    max_per_class: int | None = 1200


def iter_image_files(folder: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for file_path in folder.glob(ext):
            yield file_path


def load_dataset(cfg: TrainConfig):
    if not cfg.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {cfg.dataset_dir}")

    class_dirs = sorted([p for p in cfg.dataset_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {cfg.dataset_dir}")

    features = []
    labels = []
    class_names = [d.name for d in class_dirs]

    for class_idx, class_dir in enumerate(class_dirs):
        image_files = list(iter_image_files(class_dir))
        if cfg.max_per_class is not None:
            image_files = image_files[: cfg.max_per_class]

        for image_path in image_files:
            try:
                img = Image.open(image_path).convert("RGB")
                arr = normalize_image(img, IMG_SIZE)
                feat = extract_features(arr)
                features.append(feat)
                labels.append(class_idx)
            except Exception:
                continue

    if not features:
        raise RuntimeError("No valid images were loaded from dataset")

    X = np.vstack(features)
    y = np.array(labels, dtype=np.int64)
    return X, y, class_names


def main():
    cfg = TrainConfig()
    print(f"Loading dataset from: {cfg.dataset_dir}")
    X, y, class_names = load_dataset(cfg)

    print(f"Total samples: {len(y)}")
    print(f"Classes: {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=class_names))

    bundle = {
        "model_type": "sklearn_random_forest",
        "model": clf,
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "feature_version": 1,
    }
    joblib.dump(bundle, cfg.output_path)
    print(f"Saved fallback model to: {cfg.output_path.resolve()}")


if __name__ == "__main__":
    main()
