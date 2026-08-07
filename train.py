"""
Training script for the Physics-Informed ConvLSTM climate model.
"""

import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import ClimateNetCDFDataset
from models.convlstm import ConvLSTMForecaster
from models.pinn_loss import PhysicsInformedLoss, AdaptiveLambdaScheduler
from utils.metrics import evaluate_batch


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = Config()
    set_seed(cfg.SEED)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    train_ds = ClimateNetCDFDataset(
        cfg.DATA_PATH, cfg.VARIABLE, cfg.SEQ_LEN, cfg.PRED_LEN,
        cfg.IMG_HEIGHT, cfg.IMG_WIDTH, mode="train", train_split=cfg.TRAIN_SPLIT
    )
    val_ds = ClimateNetCDFDataset(
        cfg.DATA_PATH, cfg.VARIABLE, cfg.SEQ_LEN, cfg.PRED_LEN,
        cfg.IMG_HEIGHT, cfg.IMG_WIDTH, mode="val", train_split=cfg.TRAIN_SPLIT
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                               num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS)

    model = ConvLSTMForecaster(
        cfg.INPUT_CHANNELS, cfg.HIDDEN_DIMS, cfg.KERNEL_SIZE,
        cfg.NUM_LAYERS, cfg.PRED_LEN
    ).to(cfg.DEVICE)

    criterion = PhysicsInformedLoss(
        cfg.DIFFUSION_COEFF, cfg.DX, cfg.DY, cfg.DT
    )
    lambda_scheduler = AdaptiveLambdaScheduler(
        cfg.LAMBDA_INIT, cfg.LAMBDA_MAX, cfg.LAMBDA_WARMUP_EPOCHS, cfg.LAMBDA_RAMP_EPOCHS
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                                  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_val_rmse = float("inf")

    # --- Set up CSV logging for later visualization ---
    history_path = os.path.join(cfg.LOG_DIR, "history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lambda", "train_loss", "val_MAE", "val_RMSE", "val_physical_violation"])

    for epoch in range(cfg.EPOCHS):
        model.train()
        lambda_physics = lambda_scheduler.get_lambda(epoch)

        train_loss_sum, n_batches = 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} (λ={lambda_physics:.3f})")

        for x, y in loop:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            loss, loss_dict = criterion(pred, y, lambda_physics)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1
            loop.set_postfix(loss=loss.item(), mse=loss_dict["mse"], phys=loss_dict["physics"])

        avg_train_loss = train_loss_sum / n_batches

        # --- Validation ---
        model.eval()
        val_metrics = {"MAE": 0.0, "RMSE": 0.0, "Physical_Violation_Rate": 0.0}
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                pred = model(x)
                metrics = evaluate_batch(pred, y, denorm_fn=val_ds.denormalize)
                for k in val_metrics:
                    val_metrics[k] += metrics[k]
                n_val += 1

        for k in val_metrics:
            val_metrics[k] /= max(n_val, 1)

        scheduler.step(val_metrics["RMSE"])

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} | "
              f"val_MAE={val_metrics['MAE']:.4f} | val_RMSE={val_metrics['RMSE']:.4f} | "
              f"phys_violation={val_metrics['Physical_Violation_Rate']:.4%}")

        # --- Log this epoch's metrics to CSV ---
        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, lambda_physics, avg_train_loss,
                val_metrics["MAE"], val_metrics["RMSE"], val_metrics["Physical_Violation_Rate"]
            ])

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt"))
            print(f"  -> Saved new best model (RMSE={best_val_rmse:.4f})")


if __name__ == "__main__":
    main()