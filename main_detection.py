"""
The code belongs to the file:
    main_detection.py

This code was simplified from the Jupyter notebook on GitHub:
    ...(main).ipynb

This script is designed to detect chart patterns in stock price data using a pre-trained CNN model.
It loads historical stock data from Yahoo Finance, processes it into sliding windows, and uses a CNN
model to predict patterns. Detected patterns are then merged using a non-maximum suppression-like approach
and visualized with matplotlib.
It can be run directly from the command line with specified parameters for ticker symbol, window size,
and confidence threshold.

Example usage:

    python main_detection.py --ticker AAPL --window 64 --threshold 0.9 --model_path path/to/model.keras

OBSERVATION:
    - path/to/model.keras should be replaced with the actual path to the trained model file.
    - make sure to have the required libraries installed:
        pip install numpy yfinance matplotlib tensorflow
"""
import argparse
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- Load and normalize Yahoo Finance data ---
def load_yahoo_series(ticker_symbol="AAPL", period="6mo", interval="1d"):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period, interval=interval)
    close_series = hist["Close"].values.astype(np.float32)
    normalized = close_series / np.mean(close_series)
    return normalized, hist.index

# --- Create sliding windows ---
def sliding_windows(series, window_size=64, stride=1):
    X = []
    indices = []
    for i in range(0, len(series) - window_size + 1, stride):
        X.append(series[i:i + window_size])
        indices.append((i, i + window_size))
    return np.array(X), indices

# --- Merge overlapping detections (NMS-like) ---
def merge_boxes(boxes, iou_threshold=0.5):
    merged = []
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    while boxes:
        current = boxes.pop(0)
        merged.append(current)
        boxes = [b for b in boxes if (b['end'] < current['start'] or b['start'] > current['end']) or
                 (min(current['end'], b['end']) - max(current['start'], b['start'])) / (b['end'] - b['start']) < iou_threshold]
    return merged

# --- Main detection pipeline ---
def detect_patterns_on_ticker(ticker="AAPL", window_size=64, threshold=0.9, model_path=None):
    print(f"[INFO] Loading data for: {ticker}")
    series, time_index = load_yahoo_series(ticker)

    print(f"[INFO] Creating sliding windows...")
    X_windows, idx_pairs = sliding_windows(series, window_size=window_size)
    X_windows = np.expand_dims(X_windows, axis=-1)

    print(f"[INFO] Loading trained model...")
    #model = load_model(r"D:\dhbw_s3\wds23s4_fibu_chartformation\aaatemp\models\chartformation_cnn1d_v1.keras")
    model = load_model(model_path)  # Adjust path as needed

    print(f"[INFO] Predicting windows...")
    probs = model.predict(X_windows, verbose=0)
    preds = np.argmax(probs, axis=1)

    boxes = []
    for i, (start, end) in enumerate(idx_pairs):
        conf = np.max(probs[i])
        class_id = preds[i]
        if conf >= threshold:
            boxes.append({
                "start": start,
                "end": end,
                "class": class_id,
                "conf": conf
            })

    print(f"[INFO] Found {len(boxes)} raw boxes, applying NMS...")
    merged_boxes = merge_boxes(boxes)

    print(f"[INFO] Final detections: {len(merged_boxes)}")
    for b in merged_boxes:
        print(f"→ Class {b['class']} | {time_index[b['start']]} to {time_index[b['end']]} | Conf: {b['conf']:.2f}")

    # Plot with highlights
    plt.figure(figsize=(14, 5))
    plt.plot(series, color='black')
    for b in merged_boxes:
        plt.axvspan(b['start'], b['end'], alpha=0.3, label=f"Class {b['class']} ({b['conf']:.2f})")
    plt.title(f"Chartformations Detected on {ticker}")
    plt.xlabel("Time Index")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    detect_patterns_on_ticker(
        ticker=args.ticker,
        window_size=args.window,
        threshold=args.threshold,
        model_path=args.model_path
    )
