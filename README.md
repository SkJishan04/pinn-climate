<div align="center">

# 🌊 Physics-Informed ConvLSTM for Satellite Climate Prediction

**Constraining deep learning with physical laws for more trustworthy climate forecasts**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-NOAA%20OISST-orange)](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)

*A hybrid deep learning system that predicts sea surface temperature from satellite time-series data — constrained by real physics (advection-diffusion PDEs) so the model can't predict physically impossible outcomes.*

<!-- 🖼️ PLACEHOLDER: Hero/banner image (AI-generated) -->
<!-- ![Project Banner](assets/banner.png) -->

</div>

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Problem & Motivation](#-problem--motivation)
3. [Features](#-features)
4. [Architecture & Workflow](#-architecture--workflow)
5. [Tech Stack](#-tech-stack)
6. [Project Structure](#-project-structure)
7. [Setup & Usage](#-setup--usage)
8. [Examples](#-examples)
9. [Results & Evaluation](#-results--evaluation)
10. [Testing](#-testing)
11. [Limitations](#-limitations)
12. [Lessons Learned](#-lessons-learned)
13. [Future Improvements / Roadmap](#-future-improvements--roadmap)
14. [License & Acknowledgments](#-license--acknowledgments)

---

## 🎯 Overview

Purely data-driven CNNs and ConvLSTMs are powerful at spatiotemporal pattern matching, but they have no built-in concept of physical law. Left unconstrained, they can predict outcomes that violate basic physics — negative rainfall, spontaneous energy creation, or discontinuous jumps that no real fluid system would produce.

This project injects a physical constraint directly into the training loss of a ConvLSTM forecasting model trained on **real NOAA satellite sea surface temperature (SST) data**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \, \mathcal{L}_{\text{physics}}$$

where $\mathcal{L}_{\text{physics}}$ penalizes violations of a 2D advection-diffusion PDE, and $\lambda$ is **adaptively scheduled** during training — starting at zero so the model first learns basic data patterns, then ramping up only once training has stabilized.

The project includes a full **ablation study** comparing this physics-informed model against a pure-MSE baseline, with results reported honestly — including a tradeoff, not just a win.

---

## ❓ Problem & Motivation

**The problem:** Standard deep learning forecasters optimize purely for pointwise accuracy (MSE/MAE). This means nothing stops them from producing predictions that are statistically close to correct but physically nonsensical — a serious issue for downstream systems (e.g. hydrology models, energy balance calculations) that assume physical laws hold.

**The motivation:** Physics-Informed Neural Networks (PINNs) address this by embedding domain knowledge — in this case, a PDE governing heat/mass transport — directly into the loss function. This project explores:

- Whether a physics-constrained loss measurably reduces physically implausible predictions on **real satellite data** (not synthetic toy data).
- What the actual **accuracy tradeoff** looks like when physics constraints are added.
- The practical engineering challenges of combining PDE residuals with standard deep learning losses (which turned out to include a real numerical bug — see [Lessons Learned](#-lessons-learned)).

---

