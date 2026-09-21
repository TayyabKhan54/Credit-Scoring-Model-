"""Train script for emotion recognition using precomputed MFCC features (.npz).

Usage:
  python src/emotion_recognition/train.py --features data/emotion_features.npz
"""
import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .model import build_cnn_lstm, prepare_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    X, y, labels = prepare_data(args.features)
    # X shape is (n, time, n_mfcc)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = build_cnn_lstm(input_shape=X_train.shape[1:], n_classes=y.shape[1])
    model.summary()

    callbacks = []
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    model_path = Path(args.out_dir) / "emotion_cnn_lstm.h5"
    model.save(model_path)
    print(f"Saved trained model to {model_path}")
    # Save label mapping
    labels_path = Path(args.out_dir) / "labels.npy"
    np.save(labels_path, labels)
    print(f"Saved labels to {labels_path}")


if __name__ == "__main__":
    main()
