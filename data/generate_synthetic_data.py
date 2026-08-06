"""
Generates a synthetic NetCDF dataset that mimics NOAA OISST-style
sea surface temperature data, so the pipeline can be run end-to-end
without needing a real satellite data download first.

Simulates a diffusing "heat blob" moving across a grid over time —
structurally similar enough to SST fields that the ConvLSTM + physics
loss have real spatiotemporal patterns to learn from.
"""

import numpy as np
import xarray as xr
import argparse


def generate_synthetic_sst(time_steps=200, height=64, width=64, seed=42):
    rng = np.random.default_rng(seed)

    data = np.zeros((time_steps, height, width), dtype=np.float32)

    # Moving Gaussian blob (simulates a warm current) + diffusion + noise
    cx, cy = width * 0.3, height * 0.3
    vx, vy = 0.15, 0.1  # drift velocity per timestep
    sigma = 8.0

    yy, xx = np.mgrid[0:height, 0:width]

    for t in range(time_steps):
        cx_t = (cx + vx * t) % width
        cy_t = (cy + vy * t) % height

        blob = 15.0 * np.exp(-(((xx - cx_t) ** 2 + (yy - cy_t) ** 2) / (2 * sigma ** 2)))

        # base temperature field + seasonal-like oscillation + blob + noise
        base = 20.0 + 3.0 * np.sin(2 * np.pi * t / 50)
        noise = rng.normal(0, 0.3, size=(height, width))

        data[t] = base + blob + noise

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="./data/raw/noaa_sst.nc")
    parser.add_argument("--time_steps", type=int, default=200)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    data = generate_synthetic_sst(args.time_steps, args.height, args.width)

    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), data)},
        coords={
            "time": np.arange(data.shape[0]),
            "lat": np.linspace(-10, 10, args.height),
            "lon": np.linspace(-10, 10, args.width),
        },
    )
    ds.to_netcdf(args.out)
    print(f"Synthetic SST dataset written to {args.out}  shape={data.shape}")


if __name__ == "__main__":
    main()