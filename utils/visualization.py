"""
Plotting utilities for the Physics-Informed ConvLSTM climate model.

Two main use cases:
1. Compare predicted vs. ground-truth spatial fields (per forecast timestep).
2. Plot training/validation curves (loss, MAE, RMSE, lambda schedule) over epochs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_prediction_grid(input_seq, true_seq, pred_seq, save_path, cmap="RdYlBu_r"):
    """
    input_seq: (seq_len, H, W)   -- last few input frames shown for context
    true_seq:  (pred_len, H, W)  -- ground truth future frames
    pred_seq:  (pred_len, H, W)  -- model predictions
    Produces a 3-row grid: [Ground Truth] / [Prediction] / [Absolute Error]
    """
    pred_len = true_seq.shape[0]
    error_seq = np.abs(true_seq - pred_seq)

    vmin = min(true_seq.min(), pred_seq.min())
    vmax = max(true_seq.max(), pred_seq.max())
    err_max = error_seq.max() + 1e-6

    fig, axes = plt.subplots(3, pred_len, figsize=(3 * pred_len, 8))
    if pred_len == 1:
        axes = axes.reshape(3, 1)

    row_titles = ["Ground Truth", "Prediction", "Absolute Error"]

    for t in range(pred_len):
        im0 = axes[0, t].imshow(true_seq[t], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, t].set_title(f"t+{t+1}")
        axes[0, t].axis("off")

        im1 = axes[1, t].imshow(pred_seq[t], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, t].axis("off")

        im2 = axes[2, t].imshow(error_seq[t], cmap="hot", vmin=0, vmax=err_max)
        axes[2, t].axis("off")

    for row in range(3):
        axes[row, 0].set_ylabel(row_titles[row], fontsize=11)
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01, label="Value")
    fig.colorbar(im2, ax=axes[2, :].tolist(), fraction=0.02, pad=0.01, label="|Error|")

    plt.suptitle("Predicted vs. Ground Truth Field Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prediction comparison to {save_path}")


def plot_training_history(history, save_path):
    """
    history: dict with keys 'epoch', 'lambda', 'train_loss', 'val_MAE',
             'val_RMSE', 'val_physical_violation' -> lists
    """
    epochs = history["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Train loss
    axes[0, 0].plot(epochs, history["train_loss"], color="tab:blue")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3)

    # Lambda schedule (twin axis showing when physics kicks in)
    ax_lambda = axes[0, 0].twinx()
    ax_lambda.plot(epochs, history["lambda"], color="tab:red", linestyle="--", alpha=0.6)
    ax_lambda.set_ylabel("λ (physics weight)", color="tab:red")
    ax_lambda.tick_params(axis="y", labelcolor="tab:red")

    # MAE / RMSE
    axes[0, 1].plot(epochs, history["val_MAE"], label="Val MAE", color="tab:green")
    axes[0, 1].plot(epochs, history["val_RMSE"], label="Val RMSE", color="tab:orange")
    axes[0, 1].set_title("Validation Error")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Physical violation rate
    axes[1, 0].plot(epochs, history["val_physical_violation"], color="tab:purple")
    axes[1, 0].set_title("Physical Violation Rate (e.g. negative rainfall/energy)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Fraction of violating pixels")
    axes[1, 0].grid(alpha=0.3)

    # Lambda schedule alone (clearer view)
    axes[1, 1].plot(epochs, history["lambda"], color="tab:red")
    axes[1, 1].set_title("Adaptive λ (Physics Loss Weight) Schedule")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("λ")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training history plot to {save_path}")