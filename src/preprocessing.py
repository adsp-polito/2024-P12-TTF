"""The preprocessing module contains functions to preprocess the input data for the models."""
import pandas as pd
from src.constants import (
    SAVGOL_POLYORDER, SAVGOL_WINDOW,
    WINDOW_SIZE, WINDOW_STEP,
    SCALER_PATH, FEATURES_TO_REMOVE
)

def remove_features(df: pd.DataFrame, features_to_remove: list[str] = FEATURES_TO_REMOVE):
    return df.drop(columns=features_to_remove)

def denoise(df: pd.DataFrame, window: int = SAVGOL_WINDOW, polyorder: int = SAVGOL_POLYORDER):
    from scipy.signal import savgol_filter
    for col in df.columns:
        if col not in ['TrajectoryID', 'Cycle', 'RUL']:
            df[col] = df.groupby('TrajectoryID')[col].transform(lambda x: savgol_filter(x, window, polyorder))

    return df

def scale(df: pd.DataFrame, fit: bool = False, scaler_path: str = SCALER_PATH) -> pd.DataFrame:
    import joblib
    import os
    from sklearn.preprocessing import MinMaxScaler

    X = df.drop(columns=['TrajectoryID', 'Cycle', 'RUL'])

    if os.path.isfile(scaler_path):
        scaler: MinMaxScaler = joblib.load(scaler_path)
    else:
        scaler = MinMaxScaler()
    
    if fit:
        scaled_X = scaler.fit_transform(X)
    else:
        scaled_X = scaler.transform(X)

    scaled_df = pd.DataFrame(scaled_X, columns=X.columns, index=df.index)
    scaled_df[['TrajectoryID', 'Cycle', 'RUL']] = df[['TrajectoryID', 'Cycle', 'RUL']]

    joblib.dump(scaler, scaler_path)

    return scaled_df

def windowing(df: pd.DataFrame, window_size: int = WINDOW_SIZE, step: int = WINDOW_STEP):
    """"""
    import numpy as np

    # Prepare tensor data
    aggregate_windows = []
    corresponding_ruls = []
    trajectory_ids = []
    corresponding_cycles = []

    for trajectory_id, group in df.groupby('TrajectoryID'):
        if len(group) >= window_size: # We have to discard too short windows.
            for i in range((len(group) - window_size) % step, len(group) - window_size + 1, step):
                window = group.iloc[i:i + window_size]
                matrix = window.drop(columns=['TrajectoryID', 'Cycle', 'RUL']).values
                aggregate_windows.append(matrix)
                corresponding_ruls.append(window.iloc[-1]['RUL'])
                trajectory_ids.append(trajectory_id)
                corresponding_cycles.append(window.iloc[-1]['Cycle'])

    return np.array(aggregate_windows), np.array(corresponding_ruls), np.array(trajectory_ids), np.array(corresponding_cycles)
