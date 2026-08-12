"""
Central configuration for the Physics-Informed ConvLSTM climate model.
"""

import torch

class Config:
    # --- Data ---
    DATA_PATH = "./data/raw/noaa_sst_real.nc"     # real NOAA OISST data
    VARIABLE = "sst"                              # sea surface temperature, or "precip"
    SEQ_LEN = 10                                  # input timesteps
    PRED_LEN = 5                                  # output timesteps to forecast
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    TRAIN_SPLIT = 0.8
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    # --- Model ---
    INPUT_CHANNELS = 1
    HIDDEN_DIMS = [32, 64, 64]
    KERNEL_SIZE = (3, 3)
    NUM_LAYERS = 3

    # --- Physics ---
    # Diffusion coefficient for advection-diffusion PDE constraint
    DIFFUSION_COEFF = 0.01
    DX = 1.0
    DY = 1.0
    DT = 1.0

    # --- Adaptive Loss Weighting ---
    LAMBDA_INIT = 0.0        # start with pure data-driven loss
    LAMBDA_MAX = 0.3
    LAMBDA_WARMUP_EPOCHS = 15   # epochs before physics loss kicks in
    LAMBDA_RAMP_EPOCHS = 20     # epochs over which lambda ramps to max

    # --- Training ---
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"

    SEED = 42