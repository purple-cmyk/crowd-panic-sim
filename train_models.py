from config.settings import FEATURE_CSV_PATH
from src.ml.dataset import load_dataset
from src.ml.trainer import train_all_models


def main():
    X, y = load_dataset(FEATURE_CSV_PATH)

    if len(X) < 20:
        print("Not enough feature rows for training.")
        print("Run run_video.py first with a longer video.")
        return

    train_all_models(X, y)


if __name__ == "__main__":
    main()