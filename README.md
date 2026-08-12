<div align="center">

# 🌊 Physics-Informed ConvLSTM for Satellite Climate Prediction

**Constraining deep learning with physical laws for more trustworthy climate forecasts**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-NOAA%20OISST-orange)](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)

*A hybrid deep learning system that predicts sea surface temperature from satellite time-series data — constrained by real physics (advection-diffusion PDEs) so the model can't predict physically impossible outcomes.*

</div>

---

## 🎯 Overview

Purely data-driven CNNs and ConvLSTMs are powerful at pattern matching, but they have no concept of physical law. Left unconstrained, they can predict outcomes that violate basic physics — negative rainfall, spontaneous energy creation, or discontinuous jumps that no real fluid system would produce.

This project injects a physical constraint directly into the training loss of a ConvLSTM forecasting model:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \, \mathcal{L}_{\text{physics}}$$

where $\mathcal{L}_{\text{physics}}$ penalizes violations of a 2D advection-diffusion PDE, and $\lambda$ is **adaptively scheduled** during training — starting at zero (so the model first learns basic data patterns) and ramping up only once training has stabilized (so physics acts as a refinement, not a distraction).

The model is trained and evaluated on **real NOAA OISST satellite sea surface temperature data**, with a full ablation study comparing the physics-informed model against a pure-MSE baseline.