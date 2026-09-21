"""Smoke test: generate synthetic feature arrays and validate prepare_data.

This test does NOT require TensorFlow; it creates random MFCC-like arrays
and verifies `prepare_data` and saving/loading of .npz feature files.
"""
import numpy as np
from pathlib import Path
from .model import prepare_data


def create_fake_features(out_path: str, n_samples=50, time=174, n_mfcc=40):
    rng = np.random.default_rng(42)
    # create two classes: happy, sad
    labels = []
    X = rng.normal(0, 1, size=(n_samples, time, n_mfcc)).astype(np.float32)
    y = []
    for i in range(n_samples):
        if i < n_samples // 2:
            y.append("happy")
        else:
            y.append("sad")

    y = np.array(y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Wrote fake features to {out_path}")


def main():
    out = "data/smoke_features.npz"
    create_fake_features(out)
    X, y_cat, labels = prepare_data(out)
    print("Loaded features shapes:", X.shape, y_cat.shape, labels)


if __name__ == "__main__":
    main()
