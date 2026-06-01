import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .data_loader import FEATURE_NAMES, ZERO_INVALID_COLS, TARGET


def handle_zeros(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ZERO_INVALID_COLS:
        df[col] = df[col].replace(0, np.nan)
    return df


def impute_by_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ZERO_INVALID_COLS:
        for cls in [0, 1]:
            mask_class = df[TARGET] == cls
            median_val = df.loc[mask_class, col].median()
            missing_mask = mask_class & df[col].isna()
            df.loc[missing_mask, col] = median_val
    return df


def preprocess(df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = True):
    df = handle_zeros(df)
    df = impute_by_class(df)

    X = df[FEATURE_NAMES].values.astype(float)
    y = df[TARGET].values.astype(int)

    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def get_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = handle_zeros(df)
    return impute_by_class(df)
