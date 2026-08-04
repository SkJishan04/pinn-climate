# Physics-Informed ConvLSTM for Satellite Climate Prediction

Predicts sea surface temperature / precipitation fields from satellite
time series (NOAA / Copernicus data) using a ConvLSTM constrained by a
physics-informed loss (advection-diffusion PDE residual + non-negativity
constraint), with adaptive lambda weighting so the model learns basic
data patterns before physics constraints are enforced.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
1. Place a NetCDF file (NOAA OISST or Copernicus CMEMS) at the path set
   in `config.py` (`DATA_PATH`).
2. Train:
```bash
   python train.py
```
3. Evaluate:
```bash
   python evaluate.py
```

## How the physics constraint works
The loss combines:
- **Data loss**: standard MSE between predicted and ground-truth frames.
- **Physics loss**: residual of the 2D advection-diffusion PDE computed
  via finite differences on the predicted sequence.
- **Non-negativity penalty**: penalizes physically impossible negative
  values (e.g. rainfall, energy).

`lambda` (the physics weight) starts near 0 and ramps up after a warmup
period (`LAMBDA_WARMUP_EPOCHS`), so gradients from the physics term don't
destabilize early training.