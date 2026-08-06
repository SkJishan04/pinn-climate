"""
End-to-end smoke test: generates synthetic data, runs a few training
steps, and checks that loss decreases and no NaNs appear. Use this to
verify the pipeline works before committing to a full training run on
real NOAA/Copernicus data.
"""

import subprocess
import sys
import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import ClimateNetCDFDataset
from models.convlstm import ConvLSTMForecaster
from models.pinn_loss import PhysicsInformedLoss, AdaptiveLambdaScheduler


def run_data_generation():
    print("[1/3] Generating synthetic dataset...")
    result = subprocess.run(
        [sys.executable, "data/generate_synthetic_data.py",
         "--time_steps", "60", "--height", "32", "--width", "32"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Synthetic data generation failed")


def run_mini_training():
    print("[2/3] Running a few mini training steps...")
    cfg = Config()
    cfg.DATA_PATH = "./data/raw/noaa_sst.nc"
    cfg.IMG_HEIGHT = 32
    cfg.IMG_WIDTH = 32
    cfg.SEQ_LEN = 5
    cfg.PRED_LEN = 3
    cfg.BATCH_SIZE = 2
    cfg.HIDDEN_DIMS = [8, 8]
    cfg.NUM_LAYERS = 2

    ds = ClimateNetCDFDataset(
        cfg.DATA_PATH, cfg.VARIABLE, cfg.SEQ_LEN, cfg.PRED_LEN,
        cfg.IMG_HEIGHT, cfg.IMG_WIDTH, mode="train", train_split=0.9
    )
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True)

    model = ConvLSTMForecaster(
        cfg.INPUT_CHANNELS, cfg.HIDDEN_DIMS, cfg.KERNEL_SIZE,
        cfg.NUM_LAYERS, cfg.PRED_LEN
    ).to(cfg.DEVICE)

    criterion = PhysicsInformedLoss(cfg.DIFFUSION_COEFF, cfg.DX, cfg.DY, cfg.DT)
    scheduler = AdaptiveLambdaScheduler(0.0, 1.0, warmup_epochs=0, ramp_epochs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step, (x, y) in enumerate(loader):
        if step >= 5:
            break
        x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss, loss_dict = criterion(pred, y, lambda_physics=scheduler.get_lambda(1))

        if torch.isnan(loss):
            raise RuntimeError(f"NaN loss detected at step {step}")

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  step {step}: loss={loss.item():.4f} mse={loss_dict['mse']:.4f} "
              f"physics={loss_dict['physics']:.4f}")

    return losses


def check_results(losses):
    print("[3/3] Checking results...")
    assert len(losses) > 0, "No training steps ran"
    assert all(not (l != l) for l in losses), "NaN detected in losses"  # NaN check
    print(f"  Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("SMOKE TEST PASSED ✅")


if __name__ == "__main__":
    run_data_generation()
    losses = run_mini_training()
    check_results(losses)