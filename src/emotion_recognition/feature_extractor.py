import numpy as np
import librosa


def extract_mfcc(path, sr=22050, n_mfcc=40, max_len=174):
    """Load an audio file and return a fixed-size MFCC array.

    Returns shape (max_len, n_mfcc)
    """
    y, _ = librosa.load(path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # transpose to (time, n_mfcc)
    mfcc = mfcc.T

    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_len]

    return mfcc.astype(np.float32)


def extract_mfcc_batch(file_paths, sr=22050, n_mfcc=40, max_len=174):
    feats = [extract_mfcc(p, sr=sr, n_mfcc=n_mfcc, max_len=max_len) for p in file_paths]
    return np.stack(feats, axis=0)
