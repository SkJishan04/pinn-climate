"""
Standalone evaluation script: loads a trained checkpoint and reports
MAE / RMSE / physical-violation-rate on the held-out validation set.
"""

import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import ClimateNetCDFDataset
from models.convlstm import ConvLSTMForecaster
from utils.metrics import evaluate_batch


def main():
    cfg = Config()

    val_ds = ClimateNetCDFDataset(
        cfg.DATA_PATH, cfg.VARIABLE, cfg.SEQ_LEN, cfg.PRED_LEN,
        cfg.IMG_HEIGHT, cfg.IMG_WIDTH, mode="val", train_split=cfg.TRAIN_SPLIT
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = ConvLSTMForecaster(
        cfg.INPUT_CHANNELS, cfg.HIDDEN_DIMS, cfg.KERNEL_SIZE,
        cfg.NUM_LAYERS, cfg.PRED_LEN
    ).to(cfg.DEVICE)

    checkpoint_path = f"{cfg.CHECKPOINT_DIR}/best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.DEVICE))
    model.eval()

    totals = {"MAE": 0.0, "RMSE": 0.0, "Physical_Violation_Rate": 0.0}
    n = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            pred = model(x)
            metrics = evaluate_batch(pred, y, denorm_fn=val_ds.denormalize)
            for k in totals:
                totals[k] += metrics[k]
            n += 1

    print("=== Final Evaluation ===")
    for k in totals:
        print(f"{k}: {totals[k] / n:.4f}")


if __name__ == "__main__":
    main()