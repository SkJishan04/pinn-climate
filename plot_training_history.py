"""
Reads the training history CSV produced by train.py and generates
plots of loss, validation MAE/RMSE, physical violation rate, and the
adaptive lambda schedule over training.
"""

import os
import csv

from config import Config
from utils.visualization import plot_training_history


def load_history_csv(path):
    history = {
        "epoch": [], "lambda": [], "train_loss": [],
        "val_MAE": [], "val_RMSE": [], "val_physical_violation": []
    }
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history["epoch"].append(int(row["epoch"]))
            history["lambda"].append(float(row["lambda"]))
            history["train_loss"].append(float(row["train_loss"]))
            history["val_MAE"].append(float(row["val_MAE"]))
            history["val_RMSE"].append(float(row["val_RMSE"]))
            history["val_physical_violation"].append(float(row["val_physical_violation"]))
    return history


def main():
    cfg = Config()
    history_path = os.path.join(cfg.LOG_DIR, "history.csv")

    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"No history file found at {history_path}. Run train.py first."
        )

    history = load_history_csv(history_path)

    os.makedirs("./outputs", exist_ok=True)
    save_path = "./outputs/training_history.png"

    plot_training_history(history, save_path)


if __name__ == "__main__":
    main()