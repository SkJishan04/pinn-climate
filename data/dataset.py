"""
Dataset loader for NOAA/Copernicus NetCDF climate data.
Produces (input_sequence, target_sequence) pairs for sequence-to-sequence
spatiotemporal forecasting.
"""

import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


class ClimateNetCDFDataset(Dataset):
    def __init__(self, nc_path, variable, seq_len, pred_len,
                 img_height, img_width, mode="train", train_split=0.8):
        """
        Args:
            nc_path: path to .nc file (NOAA OISST, Copernicus CMEMS, etc.)
            variable: variable name inside the NetCDF (e.g. 'sst', 'precip')
        """
        if not os.path.exists(nc_path):
            raise FileNotFoundError(
                f"Data file not found at '{nc_path}'.\n"
                f"If you're using synthetic data, run:\n"
                f"  python data/generate_synthetic_data.py\n"
                f"first to create it, or update DATA_PATH in config.py "
                f"to point to your real NOAA/Copernicus .nc file."
            )

        ds = xr.open_dataset(nc_path)
        data = ds[variable].values.astype(np.float32)  # shape: (time, H, W)

        # Basic cleaning: fill NaNs (land masks) with local mean
        data = np.nan_to_num(data, nan=np.nanmean(data))

        # Resize/crop to fixed spatial dims if needed
        data = self._resize_spatial(data, img_height, img_width)

        # Normalize (store stats for de-normalization at inference)
        self.mean = data.mean()
        self.std = data.std() + 1e-8
        data = (data - self.mean) / self.std

        total_len = data.shape[0]
        split_idx = int(total_len * train_split)

        if mode == "train":
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]

        self.seq_len = seq_len
        self.pred_len = pred_len

        min_required = seq_len + pred_len
        if len(self.data) < min_required + 1:
            raise ValueError(
                f"Not enough timesteps for mode='{mode}': got {len(self.data)}, "
                f"need at least {min_required + 1} (SEQ_LEN={seq_len} + PRED_LEN={pred_len} + 1).\n"
                f"Either regenerate data with more time_steps, or reduce SEQ_LEN/PRED_LEN "
                f"in config.py, or increase TRAIN_SPLIT if this is the validation set."
            )

    @staticmethod
    def _resize_spatial(data, h, w):
        from scipy.ndimage import zoom
        _, cur_h, cur_w = data.shape
        if (cur_h, cur_w) == (h, w):
            return data
        zoom_factors = (1, h / cur_h, w / cur_w)
        return zoom(data, zoom_factors, order=1)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]

        # Shape: (seq_len, C=1, H, W)
        x = torch.from_numpy(x).unsqueeze(1)
        y = torch.from_numpy(y).unsqueeze(1)
        return x, y

    def denormalize(self, tensor):
        return tensor * self.std + self.mean