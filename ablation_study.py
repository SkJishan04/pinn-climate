"""
Ablation study: trains a baseline model (MSE loss only) and the
physics-informed model (MSE + adaptive physics loss) on identical data
and splits, then compares final MAE/RMSE/physical-violation-rate.

This produces the core evidence for the project's central claim:
physics-informed constraints improve accuracy over a purely data-driven
model, especially on physically implausible edge-case predictions.

Output: ./outputs/ablation_results.csv and ./outputs/ablation_comparison.png
"""

import os
import csv
import copy
import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import ClimateNetCDFDataset
from models.convlstm import ConvLSTMForecaster
from models.pinn_loss import PhysicsInformedLoss, AdaptiveLambdaScheduler
from utils.metrics import evaluate_batch
from utils.visualization import plot_ablation_comparison


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_model(cfg, train_loader, val_loader, val_ds, use_physics, tag):
    """
    Trains a single model configuration.
    use_physics=False -> lambda is always 0 (pure MSE baseline)
    use_physics=True  -> adaptive lambda schedule (your full PINN)
    """
    set_seed(cfg.SEED)  # same init for fair comparison

    model = ConvLSTMForecaster(
        cfg.INPUT_CHANNELS, cfg.HIDDEN_DIMS, cfg.KERNEL_SIZE,
        cfg.NUM_LAYERS, cfg.PRED_LEN
    ).to(cfg.DEVICE)

    criterion = PhysicsInformedLoss(cfg.DIFFUSION_COEFF, cfg.DX, cfg.DY, cfg.DT)

    if use_physics:
        lambda_scheduler = AdaptiveLambdaScheduler(
            cfg.LAMBDA_INIT, cfg.LAMBDA_MAX, cfg.LAMBDA_WARMUP_EPOCHS, cfg.LAMBDA_RAMP_EPOCHS
        )
    else:
        lambda_scheduler = None  # stays at 0 the whole time

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                                  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_val_rmse = float("inf")
    best_state = None

    print(f"\n{'='*60}\nTraining [{tag}]  (physics={'ON' if use_physics else 'OFF'})\n{'='*60}")

    for epoch in range(cfg.EPOCHS):
        model.train()
        lambda_physics = lambda_scheduler.get_lambda(epoch) if use_physics else 0.0

        for x, y in train_loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss, _ = criterion(pred, y, lambda_physics)
            loss.backward()
            optimizer.step()

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

        if (epoch + 1) % 10 == 0 or epoch == cfg.EPOCHS - 1:
            print(f"  [{tag}] epoch {epoch+1}/{cfg.EPOCHS}: "
                  f"val_MAE={val_metrics['MAE']:.4f} val_RMSE={val_metrics['RMSE']:.4f} "
                  f"phys_violation={val_metrics['Physical_Violation_Rate']:.4%}")

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dict(val_metrics)

    # Save this model's best checkpoint to disk with a tag-specific filename
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    safe_tag = tag.lower().replace(" ", "_").replace("(", "").replace(")", "")
    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"{safe_tag}.pt")
    torch.save(best_state, checkpoint_path)
    print(f"  -> Saved best checkpoint to {checkpoint_path} (val_RMSE={best_val_rmse:.4f})")

    return best_metrics


def main():
    cfg = Config()
    os.makedirs("./outputs", exist_ok=True)

    train_ds = ClimateNetCDFDataset(
        cfg.DATA_PATH, cfg.VARIABLE, cfg.SEQ_LEN, cfg.PRED_LEN,
        cfg.IMG_HEIGHT, cfg.IMG_WIDTH, mode="train", train_split=cfg.TRAIN_SPLIT
    )
    val_ds = ClimateNetCDFDataset(
        cfg.DATA_PATH, cfg.VARIABLE, cfg.SEQ_LEN, cfg.PRED_LEN,
        cfg.IMG_HEIGHT, cfg.IMG_WIDTH, mode="val", train_split=cfg.TRAIN_SPLIT
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # --- Baseline: MSE only ---
    baseline_metrics = train_one_model(
        cfg, train_loader, val_loader, val_ds, use_physics=False, tag="Baseline (MSE-only)"
    )

    # --- Physics-informed: adaptive lambda ---
    pinn_metrics = train_one_model(
        cfg, train_loader, val_loader, val_ds, use_physics=True, tag="Physics-Informed (Adaptive λ)"
    )

    # --- Save results ---
    results_path = "./outputs/ablation_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "MAE", "RMSE", "Physical_Violation_Rate"])
        writer.writerow(["Baseline (MSE-only)", baseline_metrics["MAE"],
                          baseline_metrics["RMSE"], baseline_metrics["Physical_Violation_Rate"]])
        writer.writerow(["Physics-Informed", pinn_metrics["MAE"],
                          pinn_metrics["RMSE"], pinn_metrics["Physical_Violation_Rate"]])

    rmse_improvement = (
        (baseline_metrics["RMSE"] - pinn_metrics["RMSE"]) / baseline_metrics["RMSE"] * 100
    )
    mae_improvement = (
        (baseline_metrics["MAE"] - pinn_metrics["MAE"]) / baseline_metrics["MAE"] * 100
    )

    print(f"\n{'='*60}\nABLATION RESULTS\n{'='*60}")
    print(f"{'Model':<30} {'MAE':>10} {'RMSE':>10} {'Phys.Viol.':>12}")
    print(f"{'-'*64}")
    print(f"{'Baseline (MSE-only)':<30} {baseline_metrics['MAE']:>10.4f} "
          f"{baseline_metrics['RMSE']:>10.4f} {baseline_metrics['Physical_Violation_Rate']:>11.4%}")
    print(f"{'Physics-Informed':<30} {pinn_metrics['MAE']:>10.4f} "
          f"{pinn_metrics['RMSE']:>10.4f} {pinn_metrics['Physical_Violation_Rate']:>11.4%}")
    print(f"{'-'*64}")
    print(f"RMSE improvement: {rmse_improvement:+.2f}%")
    print(f"MAE improvement:  {mae_improvement:+.2f}%")
    print(f"\nResults saved to {results_path}")

    plot_ablation_comparison(baseline_metrics, pinn_metrics, "./outputs/ablation_comparison.png")


if __name__ == "__main__":
    main()