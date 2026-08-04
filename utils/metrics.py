"""
Evaluation metrics: MAE, RMSE, and a physical-validity check
(fraction of predicted pixels that are non-physical, e.g. negative rainfall).
"""

import numpy as np
import torch


def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def physical_violation_rate(pred, lower_bound=0.0):
    """
    Fraction of predicted values below a physically valid lower bound
    (e.g. rainfall/energy should never be negative).
    """
    violations = (pred < lower_bound).float()
    return violations.mean().item()


def evaluate_batch(pred, target, denorm_fn=None):
    if denorm_fn is not None:
        pred = denorm_fn(pred)
        target = denorm_fn(target)

    return {
        "MAE": mae(pred, target),
        "RMSE": rmse(pred, target),
        "Physical_Violation_Rate": physical_violation_rate(pred),
    }