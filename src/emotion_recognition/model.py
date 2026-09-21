import os
from typing import Tuple

import numpy as np


def _try_import_keras():
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input,
            Conv1D,
            BatchNormalization,
            Activation,
            MaxPooling1D,
            Dropout,
            Bidirectional,
            LSTM,
            Dense,
            GlobalAveragePooling1D,
        )

        return (
            Model,
            Input,
            Conv1D,
            BatchNormalization,
            Activation,
            MaxPooling1D,
            Dropout,
            Bidirectional,
            LSTM,
            Dense,
            GlobalAveragePooling1D,
        )
    except Exception:
        return (None,) * 11


def build_cnn_lstm(input_shape: Tuple[int, int], n_classes: int):
    (
        Model,
        Input,
        Conv1D,
        BatchNormalization,
        Activation,
        MaxPooling1D,
        Dropout,
        Bidirectional,
        LSTM,
        Dense,
        GlobalAveragePooling1D,
    ) = _try_import_keras()

    if Model is None:
        raise ImportError("TensorFlow / Keras is required to build the model")

    inp = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2)(x)

    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def prepare_data(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]  # (n, time, n_mfcc)
    y = data["y"]
    # encode labels
    labels, inv = np.unique(y, return_inverse=True)
    y_idx = inv

    # try to use keras to_categorical if available, else use numpy one-hot
    try:
        from tensorflow.keras.utils import to_categorical

        y_cat = to_categorical(y_idx, num_classes=len(labels))
    except Exception:
        y_cat = np.eye(len(labels))[y_idx]

    return X, y_cat, labels
