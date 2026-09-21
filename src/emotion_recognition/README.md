Emotion recognition from speech — small starter module

Quick start

- Install dependencies (see repository `requirements.txt`).
- Prepare features from a dataset (examples: RAVDESS, TESS):

```
python src/emotion_recognition/data_prep.py --dataset ravdess --input PATH/TO/RAVDESS --out data/emotion_features.npz
```

- Train a model:

```
python src/emotion_recognition/train.py --features data/emotion_features.npz --epochs 25
```

Notes
- `feature_extractor.py` extracts MFCCs and pads/truncates to a fixed length.
- `data_prep.py` contains simple loaders for RAVDESS and TESS folder layouts.
- `model.py` builds a compact Conv1D + BiLSTM model using TensorFlow/Keras.

Datasets
- RAVDESS: https://zenodo.org/records/1188976
- TESS: https://tspace.library.utoronto.ca/handle/1807/24487
