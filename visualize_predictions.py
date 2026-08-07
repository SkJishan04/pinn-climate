"""
Loads a trained checkpoint, runs inference on one validation sample,
and saves a visual comparison of ground truth vs. predicted SST fields
(plus an absolute error heatmap) as a PNG.
"""

import os
import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import ClimateNetCDFDataset
from models.convlstm import ConvLSTMForecaster
from utils.visualization import plot_prediction_grid


def main():
    cfg = Config()

    val_ds = ClimateNetCDFDataset(
        cfg.DATA_PATH, cfg.VARIABLE, cfg.SEQ_LEN, cfg.PRED_LEN,
        cfg.IMG_HEIGHT, cfg.IMG_WIDTH, mode="val", train_split=cfg.TRAIN_SPLIT
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    model = ConvLSTMForecaster(
        cfg.INPUT_CHANNELS, cfg.HIDDEN_DIMS, cfg.KERNEL_SIZE,
        cfg.NUM_LAYERS, cfg.PRED_LEN
    ).to(cfg.DEVICE)

    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.DEVICE))
    model.eval()

    x, y = next(iter(val_loader))
    x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)

    with torch.no_grad():
        pred = model(x)

    # De-normalize back to real physical units (e.g. degrees C)
    x_real = val_ds.denormalize(x).cpu().numpy()[0, :, 0]      # (seq_len, H, W)
    y_real = val_ds.denormalize(y).cpu().numpy()[0, :, 0]      # (pred_len, H, W)
    pred_real = val_ds.denormalize(pred).cpu().numpy()[0, :, 0]  # (pred_len, H, W)

    os.makedirs("./outputs", exist_ok=True)
    save_path = "./outputs/prediction_comparison.png"

    plot_prediction_grid(x_real, y_real, pred_real, save_path)


if __name__ == "__main__":
    main()