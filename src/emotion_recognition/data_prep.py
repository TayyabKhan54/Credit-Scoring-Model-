"""Data preparation helpers for emotion recognition.

Supports simple loaders for RAVDESS and TESS-like layouts and extracts MFCCs.
"""
import argparse
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .feature_extractor import extract_mfcc


# RAVDESS mapping (third field) -> emotion
RAVDESS_EMO = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def load_ravdess(root_dir):
    files = []
    labels = []
    for root, _, fnames in os.walk(root_dir):
        for f in fnames:
            if not f.lower().endswith(".wav"):
                continue
            parts = f.replace(".wav", "").split("-")
            if len(parts) >= 3:
                emo = RAVDESS_EMO.get(parts[2], None)
            else:
                emo = None
            if emo is None:
                continue
            files.append(os.path.join(root, f))
            labels.append(emo)
    return files, labels


def load_tess(root_dir):
    # TESS: directories per emotion containing wavs
    files = []
    labels = []
    for emo_name in os.listdir(root_dir):
        emo_dir = os.path.join(root_dir, emo_name)
        if not os.path.isdir(emo_dir):
            continue
        for f in os.listdir(emo_dir):
            if f.lower().endswith(".wav"):
                files.append(os.path.join(emo_dir, f))
                labels.append(emo_name.lower())
    return files, labels


def build_features(file_paths, labels, out_path, sr=22050, n_mfcc=40, max_len=174):
    X = []
    y = []
    for p, lab in tqdm(list(zip(file_paths, labels)), desc="extract"):
        try:
            mf = extract_mfcc(p, sr=sr, n_mfcc=n_mfcc, max_len=max_len)
            X.append(mf)
            y.append(lab)
        except Exception:
            # skip unreadable files
            continue

    X = np.stack(X, axis=0)
    y = np.array(y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Saved features to {out_path} (X={X.shape}, y={y.shape})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ravdess", "tess"], required=True)
    parser.add_argument("--input", required=True, help="Path to dataset root")
    parser.add_argument("--out", default="data/emotion_features.npz")
    parser.add_argument("--n_mfcc", type=int, default=40)
    parser.add_argument("--max_len", type=int, default=174)
    args = parser.parse_args()

    if args.dataset == "ravdess":
        files, labels = load_ravdess(args.input)
    else:
        files, labels = load_tess(args.input)

    build_features(files, labels, args.out, n_mfcc=args.n_mfcc, max_len=args.max_len)


if __name__ == "__main__":
    main()
