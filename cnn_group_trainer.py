import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input

def filter_group_dataset(df, group_name, max_len=128):
    subset = df[df["PatternGroup"] == group_name].copy()
    subset = subset[subset["ArrayLength"] <= max_len]
    return subset

def prepare_group_arrays(df, target_len=64):
    X_raw = df["Array"].tolist()
    y_raw = df["Pattern"].astype("category").cat.codes
    label_map = dict(enumerate(df["Pattern"].astype("category").cat.categories))
    X_padded = pad_sequences(X_raw, target_len=target_len)
    return X_padded, y_raw.to_numpy(), label_map

def build_adaptive_model(input_len, num_classes):
    model = Sequential()
    model.add(Input(shape=(input_len, 1)))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))

    if num_classes == 1:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model_per_group(df, group_name, target_len=64, model_dir="models_keras"):
    os.makedirs(model_dir, exist_ok=True)

    df_group = filter_group_dataset(df, group_name, max_len=target_len)
    if len(df_group) < 20:
        print(f"[WARN] Not enough data for group {group_name}")
        return

    X, y, label_map = prepare_group_arrays(df_group, target_len=target_len)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    model = build_adaptive_model(input_len=target_len, num_classes=len(set(y)))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=1)

    model_path = os.path.join(model_dir, f"cnn_1d_{group_name}.keras")
    model.save(model_path)
    print(f"[INFO] Model for {group_name} saved at: {model_path}")

    return model_path, label_map

def train_all_groups(df, model_dir="models_keras", target_len=64):
    for group in df["PatternGroup"].unique():
        print(f"\n[INFO] Training model for group: {group}")
        train_model_per_group(df, group, target_len=target_len, model_dir=model_dir)
