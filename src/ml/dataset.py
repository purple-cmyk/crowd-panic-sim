import pandas as pd
from config.settings import FEATURE_NAMES


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    df = df.dropna()

    X = df[FEATURE_NAMES]
    y = df["pseudo_label"]

    return X, y