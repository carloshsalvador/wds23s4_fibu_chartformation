"""
utils_test.py
This module contains unit tests for the utility functions defined in utils.py.

Use the test on the terminal with:
    pytest utils_test.py -v

"""

import numpy as np
import pandas as pd
import pytest
from utils import (
    pad_sequences, 
    split_dataset, 
    build_cnn_model, 
    get_class_distribution,
    load_npy_dataset,
    load_yahoo_series,
    sliding_windows,
    merge_boxes
)

# --- Test: pad_sequences ---
def test_pad_sequences():
    X = [np.array([1, 2]), np.array([1, 2, 3]), np.array([1])]
    padded = pad_sequences(X, target_len=4)
    assert padded.shape == (3, 4)
    assert (padded[0] == np.array([1, 2, 0, 0])).all()
    assert (padded[1] == np.array([1, 2, 3, 0])).all()

# --- Test: split_dataset ---
def test_split_dataset():
    X = np.random.rand(100, 64)
    y = np.array([0]*50 + [1]*50)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0
    assert len(X_train) + len(X_val) + len(X_test) == 100

# --- Test: build_cnn_model ---
def test_build_cnn_model():
    model = build_cnn_model(input_length=64, num_classes=4)
    assert model.input_shape == (None, 64, 1)
    assert model.output_shape[-1] == 4

# --- Test: get_class_distribution ---
def test_get_class_distribution():
    y = np.array([0]*3 + [1]*2 + [2]*1)
    dist = get_class_distribution(y)
    assert dist[0] == 3 and dist[1] == 2 and dist[2] == 1

# --- Optional Integration Tests (skip if no files) ---

def test_load_yahoo_series():
    series, index = load_yahoo_series("AAPL", period="5d")
    assert isinstance(series, np.ndarray)
    assert len(series) > 0

def test_sliding_windows():
    s = np.arange(10)
    X, idx = sliding_windows(s, window_size=5)
    assert len(X) == 6
    assert (X[0] == np.array([0,1,2,3,4])).all()


def test_merge_boxes():
    boxes = [
        {'start': 0, 'end': 10, 'conf': 0.9},
        {'start': 5, 'end': 15, 'conf': 0.8},
        {'start': 20, 'end': 30, 'conf': 0.95}
    ]
    merged = merge_boxes(boxes, iou_threshold=0.5)
    assert len(merged) == 2  # Should merge the first two
    assert any(b['start'] == 20 for b in merged)
