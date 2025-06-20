
"""
utils.py
This module contains utility functions for data extraction, transformation, and model training.

It includes functions for extracting data from a zip file, calculating time ranges, normalizing arrays,
padding sequences, loading datasets, splitting datasets, and training Keras models for different pattern groups.

It belongs to the Pattern Recognition project, which is focused on analyzing financial patterns.

https://github.com/carloshsalvador/wds23s4_fibu_chartformation



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import itertools
import zipfile
import py7zr
import os
import json
import pickle
import yfinance as yf
import tensorflow as tf
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.interpolate import interp1d


### --- ETL Raw Data ---
### --- ETL Raw Data: RHermaeus1618 ---

def dataset_extract_from_zip(zip_path):
    """
    Extracts the dataset from a zip file and returns a DataFrame with the data.

    Data from https://github.com/Hermaeus1618/PatternRecognition
    
    Args:
        zip_path (str): Path to the zip file.
    
    Returns:
        pd.DataFrame: DataFrame containing the extracted data.
    """
    ZIP_PATH = zip_path


    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"The file '{ZIP_PATH}' does not exist. Please check the file path.")


    data = []

    with zipfile.ZipFile(ZIP_PATH, "r") as zfile:
        # Ler todos os nomes de arquivos no ZIP
        filenames = [f.filename for f in zfile.infolist()]
    
        for fname in filenames:
            # Dividir o nome do arquivo em partes
            # Values are seprated by "_GAP_" string which can be easily converted back into list by pd.Series().str.split()
            # Naming convention for Numpy files: IOC_GAP_CupAndHandle_GAP_HOUR1_GAP_1722503700_GAP_1725272100 STOCK_GAP_PATTERN_GAP_TIMEFRAME_GAP_START_GAP_END
            parts = fname.split("_GAP_")
            if len(parts) != 5:
                print(f"Ignorando arquivo com formato inválido: {fname}")
                continue
            
            try:
                company, pattern, frequency, start, end = parts
                start = int(start)
                end = int(end)
            except ValueError as e:
                print(f"Erro ao converter start/end para inteiro no arquivo {fname}: {e}")
                continue
            
            # Ler os dados binários e convertê-los para um array Float64
            try:
                with zfile.open(fname) as file:
                    array = np.frombuffer(file.read(), dtype=np.float64)
                if array.size == 0:
                    print(f"Array vazio no arquivo {fname}. Ignorando.")
                    continue
            except Exception as e:
                print(f"Erro ao ler os dados binários do arquivo {fname}: {e}")
                continue
            
            # Adicionar os dados à lista
            data.append({
                "Company": company,
                "Pattern": pattern,
                "Frequency": frequency,
                "Start": start,
                "End": end,
                "Array": array
            })

    df = pd.DataFrame(data)

    return df

def calculate_time_range(row):
    """
    It is a complement to the dataset_extract_from_zip() function, that extracts the a specific data from:
    https://github.com/Hermaeus1618/PatternRecognition

    It dependes on the frequency of the data to calculate the time range, based on the start and end timestamps.
    It is used to create the time range for the data in the DataFrame.

    Simple check as follow:
    print(df["Frequency"].unique()) # ['DAILY' 'WEEKLY' 'MONTHLY' 'MIN5' 'MIN15' 'MIN30' 'HOUR1']
    ... where df is the DataFrame returned by dataset_extract_from_zip()
    
    Frequencies:
    - DAILY: Daily intervals
    - WEEKLY: Weekly intervals
    - MONTHLY: Monthly intervals
    - MIN5: 5-minute intervals
    - MIN15: 15-minute intervals

    Args:
        row (pd.Series): A row from the DataFrame.
    Returns:
        pd.DatetimeIndex: A DatetimeIndex object representing the time range.

    Example:
        df = dataset_extract_from_zip("path/to/your/file.zip")
        df["TimeRange"] = df.apply(calculate_time_range, axis=1)
        print(df[["Company", "Pattern", "Frequency", "TimeRange"]])
    """


    frequency = row["Frequency"]
    start = row["Start"]
    end = row["End"]

    # ['DAILY' 'WEEKLY' 'MONTHLY' 'MIN5' 'MIN15' 'MIN30' 'HOUR1']
    
    if frequency == "DAILY":        
        return pd.date_range(start=pd.to_datetime(start, unit="s"), 
                             end=pd.to_datetime(end, unit="s"), 
                             freq="D")
    
    elif frequency == "WEEKLY":
        return pd.date_range(start=pd.to_datetime(start, unit="s"), 
                             end=pd.to_datetime(end, unit="s"), 
                             freq="W")
    
    elif frequency == "MONTHLY":
        return pd.date_range(start=pd.to_datetime(start, unit="s"), 
                                end=pd.to_datetime(end, unit="s"), 
                                freq="M")

    elif frequency == "MIN5":
        return pd.date_range(start=pd.to_datetime(start, unit="s"), 
                             end=pd.to_datetime(end, unit="s"), 
                             freq="5T")
    
    elif frequency == "MIN15":
        return pd.date_range(start=pd.to_datetime(start, unit="s"), 
                                end=pd.to_datetime(end, unit="s"), 
                                freq="15T")
    elif frequency == "MIN30":
        return pd.date_range(start=pd.to_datetime(start, unit="s"), 
                                end=pd.to_datetime(end, unit="s"), 
                                freq="30T")
    
    elif frequency == "HOUR1":
        return pd.date_range(start=pd.to_datetime(start, unit="s"), 
                            end=pd.to_datetime(end, unit="s"), 
                            freq="H")
    else:
        print(f"Unknown frequency: {frequency}")
        return None
    

def calculate_time_range_per_row(row):
    """
    It is a complement to the dataset_extract_from_zip() function, that extracts the a specific data from:
    https://github.com/Hermaeus1618/PatternRecognition

    It dependes on the frequency of the data to calculate the time range, based on the start and end timestamps.
    It is used to create the time range for the data in the DataFrame.

    Simple check as follow:
    print(df["Frequency"].unique()) # ['DAILY' 'WEEKLY' 'MONTHLY' 'MIN5' 'MIN15' 'MIN30' 'HOUR1']
    ... where df is the DataFrame returned by dataset_extract_from_zip()
    
    Frequencies:
    - DAILY: Daily intervals
    - WEEKLY: Weekly intervals
    - MONTHLY: Monthly intervals
    - MIN5: 5-minute intervals
    - MIN15: 15-minute intervals

    Args:
        row (pd.Series): A row from the DataFrame.
    Returns:
        pd.DatetimeIndex: A DatetimeIndex object representing the time range.

    Example:
        df = dataset_extract_from_zip("path/to/your/file.zip")
        df["TimeRange"] = df.apply(calculate_time_range, axis=1)
        print(df[["Company", "Pattern", "Frequency", "TimeRange"]])
    """


    frequency = row["Frequency"]
    start = row["Start"]
    end = row["End"]

    try: # Validation
        start = int(start)
        end = int(end)        
        if start <= 0 or end <= 0 or start >= end:
            print(f"invalid values: start={start}, end={end}")
            return (None, None, None)           
    except ValueError as e:
        print(f"Error by convering start/end to int: {e}")
        return (None, None, None)
        
    start=pd.to_datetime(start, unit="s")
    end=pd.to_datetime(end, unit="s")    
    
    freq_map = {
    "DAILY": "D",
    "WEEKLY": "W",
    "MONTHLY": "ME",
    "MIN5": "5min",
    "MIN15": "15min",
    "MIN30": "30min",
    "HOUR1": "h"
    }

    freq = freq_map.get(frequency)
    
    if freq:
        range = pd.date_range(start=start, end=end, freq=freq)
        return (start, end, range)
    else:
        print(f"Unknown frequency: {frequency}")
        return (start, end, None)
    
def is_candlestick_viable(row):
    """
    Avalia se é viável gerar OHLC (candlestick) a partir dos dados de um registro.

    Condições:
    - Frequência precisa ser subdiária (MIN5, MIN15, MIN30, HOUR1)
    - TimeRange precisa conter pelo menos 2 pontos por dia útil

    Retorna:
    - True se for possível gerar OHLC válido
    - False caso contrário
    """
    frequency = row.get("Frequency")
    time_range = row.get("TimeRange")

    if time_range is None or not isinstance(time_range, pd.DatetimeIndex):
        return False

    # Frequências subdiárias
    subdaily = ["MIN5", "MIN15", "MIN30", "HOUR1"]
    if frequency not in subdaily:
        return False

    # Contagem de pontos por dia
    counts_per_day = pd.Series(time_range).groupby(lambda x: x.date()).count()

    # Verifica se há pelo menos 2 pontos por dia (mínimo para OHLC)
    return counts_per_day.ge(2).any()


def array_per_row_normalization(row):
    """
    
    """
    array = row.get("Array")
    
    if array is None or not isinstance(array, np.ndarray):
        return None

    # Normalização
    min_val = np.min(array)
    max_val = np.max(array)
    
    if max_val == min_val:
        return None  # Evita divisão por zero
    
    normalized_array = (array - min_val) / (max_val - min_val)
    
    return normalized_array 


def dataset_balanced(dataset):
    class_ref_name = dataset["Pattern"].value_counts().idxmin()
    class_ref_nmin = dataset["Pattern"].value_counts().min()
    class_ref_median = dataset[dataset["Pattern"]==class_ref_name]["ArrayLength"].median()

    dataset_sorted = dataset.copy()
    dataset_sorted["ref"] = [0 if dataset["Pattern"].iloc[x]==class_ref_name else 1 for x in range(len(dataset_sorted))]
    dataset_sorted["ref_dif"] = ((dataset_sorted["ArrayLength"] - class_ref_median)**2)**(1/2)
    dataset_sorted = dataset_sorted.sort_values(by=["ref", "Pattern", "ref_dif"])
    dataset_sorted = dataset_sorted.groupby("Pattern").head(class_ref_nmin).reset_index(drop=True)

    return dataset_sorted

def dataset_balanced_filtered(dataset, max_len=128):
    dataset_filtered = dataset[dataset["ArrayLength"] <= max_len].copy()
    class_ref_name = dataset_filtered["Pattern"].value_counts().idxmin()
    class_ref_nmin = dataset_filtered["Pattern"].value_counts().min()
    class_ref_median = dataset_filtered[dataset_filtered["Pattern"] == class_ref_name]["ArrayLength"].median()

    dataset_filtered["ref_dif"] = ((dataset_filtered["ArrayLength"] - class_ref_median) ** 2) ** 0.5
    dataset_filtered["ref"] = (dataset_filtered["Pattern"] != class_ref_name).astype(int)
    dataset_filtered = dataset_filtered.sort_values(by=["ref", "Pattern", "ref_dif"])
    dataset_filtered = dataset_filtered.groupby("Pattern").head(class_ref_nmin).reset_index(drop=True)
    return dataset_filtered

### --- Raw Data from .zip or df to plot to .npy structure ---

def plot_and_save_by_pattern_from_zipfile(zip_path, pattern_name, max_plots=None, dropbox_folder=None):
    """
    Extrahiert und speichert alle Plots für ein bestimmtes Muster aus einer ZIP-Datei.
    :param zip_path: Pfad zur ZIP-Datei
    :param pattern_name: Name des Musters, das extrahiert werden soll
    :param max_plots: Maximale Anzahl der Plots, die gespeichert werden sollen (None für alle)
    :param dropbox_folder: Zielordner für die gespeicherten Plots (None für aktuellen Ordner)

    Example:
    # Plots für die ersten 10 Dateien erstellen
        dataset_path_dropbox = r"C:\\Users\\...\\Dropbox\\wds23s4_bwl_chartformation_dataset"
        plot_and_save_all_by_pattern(ZIP_PATH, "HeadAndShoulder", max_plots=10, dropbox_folder=dataset_path_dropbox)
    """

    # Zielordner erstellen
    if dropbox_folder is not None:
        dataset_folder = os.path.join(dropbox_folder, pattern_name)
    
    os.makedirs(dataset_folder, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zfile:
        # Dateinamen und Metadaten extrahieren
        files = pd.Series([f.filename for f in zfile.infolist()])
        meta_df = files.str.split("_GAP_", expand=True)
        meta_df.columns = ["ticker", "pattern", "freq", "start_ts", "end_ts"]
        meta_df["filename"] = files

        # Filter auf gewünschtes Pattern
        subset = meta_df[meta_df["pattern"] == pattern_name]

        # Begrenzung der Anzahl der Plots
        if max_plots is not None:
            subset = subset.head(max_plots)

        for i, row in subset.iterrows():
            file = row["filename"]
            ticker = row["ticker"]
            try:
                with zfile.open(file) as f:
                    data = np.frombuffer(f.read(), dtype=np.float64)

                # Plot erstellen
                plt.figure(figsize=(8, 3))
                plt.plot(data, color="black")  # Linie in Schwarz
                plt.axis("off")  # Achsen ausblenden
                plt.tight_layout()

                # Dateiname sichern
                safe_ticker = ticker.replace("/", "_")
                filename = os.path.join(dataset_folder, f"{i:03d}_{safe_ticker}.png")
                plt.savefig(filename, bbox_inches="tight", pad_inches=0)
                plt.close()

            except Exception as e:
                print(f"Fehler bei der Verarbeitung von {file}: {e}")

def plot_and_save_by_pattern_from_df(df, pattern_name, max_plots=None, dropbox_folder=None, img_size=(224, 224), dpi=100):
    """
    Extraction and saving of all plots for a specific pattern from a Pandas DataFrame.
  
    The image are prepared to CNN models keeping the aspect ratio of 1:1 and the size of 224x224 pixels, it helps to train the model afterwards.
    The images are saved in the format: {i:04d}_{ticker}.png per folder {pattern_name}.

    Drobox folder is used to save the images, and share the results with the team.

    This function follows some key points for CNN models:
        * Always use fixed size images
        * Normalize either during image generation or via preprocessing layer. Use the `Array_norm` instead of Array!
        * Balance the number of samples per class (1:1 ratio) to avoid class bias
        * Save datasets in folder structure: dataset/ClassName/*.png
        * Start simple (e.g., black line on white background) before moving to complex inputs

    Args:
        :param df: Data Frame mit den Spalten "Pattern", "Array_norm", "Company"
        :param img_size: Größe der Bilder in Pixeln (z.B. 224x224).
        :param dpi: Auflösung der Bilder 
        :param pattern_name: Name des Musters, das extrahiert werden soll
        :param max_plots: Maximale Anzahl der Plots, die gespeichert werden sollen (None für alle)
        :param dropbox_folder: Zielordner für die gespeicherten Plots (None für aktuellen Ordner)

    Example:
    # Plots für die ersten 10 Dateien erstellen
    dataset_path_dropbox = "C:.../Dropbox/wds23s4_bwl_chartformation_dataset"
    plot_and_save_all_by_pattern(ZIP_PATH, "HeadAndShoulder", max_plots=10, dropbox_folder=dataset_path_dropbox)

    """
    if dropbox_folder is not None:
        dataset_folder = os.path.join(dropbox_folder, pattern_name)
        os.makedirs(dataset_folder, exist_ok=True)
    else:
        dataset_folder = os.getcwd()

    subset = df[df["Pattern"] == pattern_name]
    if max_plots is not None:
        subset = subset.head(max_plots)

    for i, row in subset.iterrows():
        ticker = str(row['Company'])
        data = row["Array_norm"]

        fig = plt.figure(figsize=(img_size[0]/dpi, img_size[1]/dpi), dpi=dpi)
        plt.plot(data, color="black", linewidth=2)
        plt.axis("off")
        plt.tight_layout(pad=0)

        filename = os.path.join(dataset_folder, f"{i:04d}_{ticker}.png")
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def save_array_as_npy_by_pattern(df, pattern_name, max_files=None, output_folder=None, normalize_method="mean"):
    """
    Alternative to the plot_and_save_by_pattern_from_df() function, that saves time series arrays as normalized .npy files organized by pattern (class).
    The .npy files are saved in the format: {i:04d}_{ticker}.npy per folder {pattern_name}.
    This function is more efficient for training models that accept raw time series data, such as CNN models.
    It avoids the need to generate images, and is useful for training models that accept raw time series data.
    The simple plot and save as .png did by plot_and_save_by_pattern_from_df() converts pixels values to 0-255 and looses the original data range.
    Moreover, it will require more disk space, time for saving and loading, and is not suitable for models that expect raw time series data, i.e. CNN models will retransform the image back to process it.
    The only advantage of the plot_and_save_by_pattern_from_df() is that it can be used to visualize the data, but it is not necessary for training models.

    Args:
        :param df (pd.DataFrame): DataFrame containing the time series data.
        The DataFrame should have the following columns:
        - "Pattern": name pattern (class) of the time series.
        - "Array": numpy array of the time series data.
        - "Company": name of the company (optional, can be used to create unique filenames).
        The DataFrame can be created by the dataset_extract_from_zip() function, which extracts the data from a zip file.
        :param pattern_name (str): Name of the pattern (class) to filter the DataFrame.
        :param max_files (int or None): Maximum number of files to save. If None, saves all files.
        :param output_folder (str or None): Root path to save the .npy files. If None, saves in the current working directory.
        :param normalize_method (str): Normalization method to apply to the time series data. The options are:
        - "mean": Normalizes the data by dividing by the mean value.
        - "minmax": Normalizes the data to the range [0, 1] using min-max scaling.
    Returns:
        None. The function saves the .npy files in the specified output folder.
      
    Example:
    # Save the first 10 files of the "HeadAndShoulder" pattern as .npy files
    dataset_path_dropbox = "C:.../Dropbox/chartformation_dataset"
    df = dataset_extract_from_zip("path/to/your/file.zip")
    save_array_as_npy_by_pattern(df, "HeadAndShoulder", max_files=10, output_folder=dataset_path_dropbox, normalize_method="mean")

    """
    if output_folder is None:
        output_folder = os.getcwd()
    
    pattern_folder = os.path.join(output_folder, pattern_name)
    os.makedirs(pattern_folder, exist_ok=True)

    subset = df[df["Pattern"] == pattern_name]
    if max_files:
        subset = subset.head(max_files)

    for i, row in subset.iterrows():
        array = np.array(row["Array"], dtype=np.float32)
        company = str(row.get("Company", "unknown")).replace("/", "_")

        # Normalization
        if normalize_method == "mean":
            mean = np.mean(array)
            if mean != 0:
                array = array / mean
        elif normalize_method == "minmax":
            min_val, max_val = np.min(array), np.max(array)
            if max_val > min_val:
                array = (array - min_val) / (max_val - min_val)

        filename = os.path.join(pattern_folder, f"{i:04d}_{company}.npy")
        np.save(filename, array)



### --- Padding ---
def pad_sequences_old(X, target_len=None):
    """
    Pads sequences to the same length.

    Args:
        X (list of np.ndarray): List of sequences (numpy arrays) to be padded.
        target_len (int, optional): Length to which all sequences should be padded. 
                                    If None, it will use the length of the longest sequence.

    Returns:
        np.ndarray: 2D array with padded sequences.
    """
    if target_len is None:
        target_len = max(len(x) for x in X)
    padded_X = np.array([np.pad(x, (0, target_len - len(x))) for x in X])
    return padded_X

def interpolate_sequence(x, target_len, method="linear"):
    """
    Interpolates (resizes) a 1D sequence to the target length using linear interpolation.
    Args:
        x (np.ndarray): 1D array to be interpolated.
        target_len (int): Desired length of the output array.
    Returns:
        np.ndarray: Interpolated 1D array of length target_len.
    """
    if len(x) == target_len:
        return x
    
    if method == "linear":
        xp = np.linspace(0, 1, num=len(x))
        fp = x
        x_new = np.linspace(0, 1, num=target_len)
        return np.interp(x_new, xp, fp)
    
    if method == "cubic":
        original_idx = np.linspace(0, 1, len(seq))
        target_idx = np.linspace(0, 1, target_len)
        f_interp = interp1d(original_idx, seq, kind='linear')
        return f_interp(target_idx)

def pad_sequences(X, target_len=None, method="zero"):
    """
    Pads or resamples sequences to the same length.

    Args:
        X (list of np.ndarray): List of sequences (numpy arrays) to be padded.
        target_len (int, optional): Length to which all sequences should be padded. 
                                    If None, it will use the length of the longest sequence.
        method (str): "zero" for zero-padding, "interpolate" for interpolation-based resampling.

    Returns:
        np.ndarray: Array of padded sequences.
    """
    if target_len is None:
        target_len = max(len(x) for x in X)

    if method == "interpolate":
        return np.array([interpolate_sequence(x, target_len) for x in X])
    elif method == "zero":
        return np.array([np.pad(x, (0, target_len - len(x))) for x in X])
    else:
        raise ValueError("Invalid padding method. Choose 'zero' or 'interpolate'")

### --- Loading dataset from npy structure ---
def load_npy_dataset(dataset_dir):
    """
    Load .npy dataset from folder structure with one folder per class.

    Args:
        dataset_dir (str): Path to dataset folder.

    Returns:
        tuple: (X_raw, y_raw, label_map)
    """
    X_raw = []
    y_raw = []
    label_map = {}
    for idx, class_name in enumerate(sorted(os.listdir(dataset_dir))):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            label_map[idx] = class_name
            for fname in os.listdir(class_path):
                if fname.endswith(".npy"):
                    data = np.load(os.path.join(class_path, fname))
                    X_raw.append(data)
                    y_raw.append(idx)
    return X_raw, y_raw, label_map


### --- Train/Val/Test Split ---
def split_dataset(X_raw, y_raw, target_len=64, test_size=0.2, val_size=0.5, random_state=42, method = "zero"):
    """
    Pad and split the data into train, val, and test sets.

    Args:
        X_raw (list): List of 1D numpy arrays
        y_raw (list): Corresponding labels
        target_len (int): Final length after padding

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_padded = pad_sequences(X_raw, target_len=target_len, method = method)
    y = np.array(y_raw)

    # Split 1: train vs temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_padded, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Split 2: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


### --- Label name retrieval ---
def get_label_names(y_pred, label_map):
    """
    Convert numeric predictions to label names using label_map.

    Args:
        y_pred (array): Predictions (numeric)
        label_map (dict): Map from index to class name

    Returns:
        list: Class names
    """
    reverse_map = {v: k for k, v in label_map.items()}
    if isinstance(y_pred[0], str):
        return y_pred  # already strings
    return [label_map.get(i, "Unknown") for i in y_pred]



### --- Keras Model Functions ---
### --- Group Strategie ---
def filter_group_dataset(df, group_name, max_len=128):
    subset = df[df["PatternGroup"] == group_name].copy()
    subset = subset[subset["ArrayLength"] <= max_len]
    return subset

def prepare_group_arrays(df, target_len=64, method = "zero"):
    X_raw = df["Array"].tolist()
    y_raw = df["Pattern"].astype("category").cat.codes
    label_map = dict(enumerate(df["Pattern"].astype("category").cat.categories))
    X_padded = pad_sequences(X_raw, target_len=target_len, method = method)
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


def train_model_per_group(df, group_name, target_len=64, model_dir="models_keras", method="zero"):
    os.makedirs(model_dir, exist_ok=True)

    df_group = filter_group_dataset(df, group_name, max_len=target_len)
    if len(df_group) < 20:
        print(f"[WARN] Not enough data for group {group_name}")
        return

    # Balance dataset! important to avoid class bias
    df_group = dataset_balanced_filtered(df_group, max_len=target_len) 

    # tradional split! :)
    X, y, label_map = prepare_group_arrays(df_group, target_len=target_len, method=method)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # model building and training (.fit!)
    model = build_adaptive_model(input_len=target_len, num_classes=len(set(y)))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        verbose=1
    )

    # --- Save paths ---
    base_path = os.path.join(model_dir, f"cnn_1d_{group_name}")
    model_path = base_path + ".keras"
    label_map_path = base_path + "_labelmap.json"
    history_path = base_path + "_history.pkl"

    model.save(model_path)
    print(f"[INFO] Model for {group_name} saved at: {model_path}")

    with open(label_map_path, "w") as f:
        json.dump(label_map, f)
    print(f"[INFO] Label map saved at: {label_map_path}")

    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"[INFO] Training history saved at: {history_path}")

    return model_path, label_map, history.history

def train_all_groups(df, model_dir="models_keras", target_len=64):
    for group in df["PatternGroup"].unique():
        print(f"\n[INFO] Training model for group: {group}")
        train_model_per_group(df, group, target_len=target_len, model_dir=model_dir)

### --- Group Strategie: MODEL EVALUATION ---

# --- Visualization helpers ---
def plot_training_history(history_dict, title="Training History"):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history_dict["accuracy"], label="Train")
    plt.plot(history_dict["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history_dict["loss"], label="Train")
    plt.plot(history_dict["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Confusion matrix

def plot_confusion_matrix(y_true, y_pred, class_names, normalize='true', title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

#### --- Yahoo Finance Data ---
# copy from main_detection.py

# --- Load and normalize Yahoo Finance data ---
def load_yahoo_series(ticker_symbol="AAPL", period="6mo", interval="1d"):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period, interval=interval)
    close_series = hist["Close"].values.astype(np.float32)
    normalized = close_series / np.mean(close_series) # the same normalization method used in the training's dataset!
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
# Non-Maximum Suppression (NMS) [Canny, 1986]
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
def detect_patterns_on_ticker(ticker="AAPL", window_size=64, threshold=0.9, model_path=None, label_path=None, label_prefix=None):
    print(f"[INFO] Loading data for: {ticker}")
    series, time_index = load_yahoo_series(ticker)

    print(f"[INFO] Creating sliding windows...")
    X_windows, idx_pairs = sliding_windows(series, window_size=window_size)
    X_windows = np.expand_dims(X_windows, axis=-1)

    print(f"[INFO] Loading trained model...")
    model = load_model(model_path)

    # load label_map when label_path is provided
    if label_path and os.path.exists(label_path):
        with open(label_path, "r") as f:
            label_map = json.load(f)
        # to make sure... converting keys to int! 
        label_map = {int(k): v for k, v in label_map.items()}
    else:
        label_map = None

    #print("label_map:", label_map) # just for debug! 

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

    # Plot with highlights
    plt.figure(figsize=(14, 5))
    #plt.plot(series, color='black')
    plt.plot(time_index, series, color='black')
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # pattern_to_color = {}
        
    # for b in merged_boxes:
    #     start_idx = b['start']
    #     end_idx = min(b['end'], len(time_index) - 1)
    #     pattern_name = label_map[b['class']] if label_map else str(b['class'])
    #     prefix = f"{label_prefix} - " if label_prefix else ""
    #     print(f"→ {prefix}{pattern_name} | {time_index[start_idx]} to {time_index[end_idx]} | Conf: {b['conf']:.2f}")

    #     # standardize colors for patterns
    #     if pattern_name not in pattern_to_color:
    #         pattern_to_color[pattern_name] = next(colors)
    #     plt.axvspan(
    #         #start_idx, end_idx,
    #         time_index[start_idx], time_index[end_idx],
    #         alpha=0.3,
    #         color=pattern_to_color[pattern_name],
    #         label=f"{prefix}{pattern_name} ({b['conf']:.2f})"
    #     )

    handles = []

    for i, b in enumerate(merged_boxes):
        start_idx = b['start']
        end_idx = min(b['end'], len(time_index) - 1)
        pattern_name = label_map[b['class']] if label_map else str(b['class'])
        prefix = f"{label_prefix} - " if label_prefix else ""
        color = next(colors)
        label = f"{prefix}{pattern_name} ({b['conf']:.2f})"

        print(f"→ {label} | {time_index[start_idx]} to {time_index[end_idx]} | Conf: {b['conf']:.2f}")

        span = plt.axvspan(
            time_index[start_idx], time_index[end_idx],
            alpha=0.3,
            color=color,
            label=label
        )
        handles.append(mpatches.Patch(color=color, label=label))



    plt.title(f"Chartformations Detected on {ticker}")
    plt.xlabel("Time Index")
    plt.ylabel("Normalized Price")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)


    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if labels:
        plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()

def detect_all_groups_on_ticker(ticker, model_folder, window_size=64, threshold=0.9):
    for group_file in os.listdir(model_folder):
        if group_file.endswith(".keras"):
            if group_file.startswith("cnn_1d_"):
                group_name = group_file.replace("cnn_1d_", "").replace(".keras", "")
            else:
                group_name = group_file.replace(".keras", "")



            model_path = os.path.join(model_folder, group_file)
            label_path = model_path.replace(".keras", "_labelmap.json")
            detect_patterns_on_ticker(
                ticker=ticker,
                window_size=window_size,
                threshold=threshold,
                model_path=model_path,
                label_path=label_path,
                label_prefix=group_name  # imprtant here to avoid name conflicts with different groups
            )


