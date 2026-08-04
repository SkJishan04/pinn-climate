"""
Physics-Informed loss module.

Encodes a 2D advection-diffusion PDE as a soft constraint:

    dC/dt + u*(dC/dx) + v*(dC/dy) = D * (d2C/dx2 + d2C/dy2)

For climate fields like SST/precipitation, this approximates conservation
of the transported quantity under diffusion + advection. The residual of
this equation is penalized in the loss so that predictions cannot show
non-physical behavior like energy/mass creation.

Includes an AdaptiveLambdaScheduler that ramps up the physics weight
lambda only after the model has learned basic data patterns (data-first,
physics-later curriculum), which is the standard trick used to get PINNs
past the naive-solution local minimum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_gradients(field, dx, dy):
    """
    Computes first and second spatial derivatives via Sobel-like central
    differences (padded so spatial dims are preserved).
    field: (B, T, C, H, W)
    """
    # Central difference kernels
    kernel_dx = torch.tensor([[-1, 0, 1]], dtype=field.dtype, device=field.device) / (2 * dx)
    kernel_dy = kernel_dx.transpose(0, 1)

    kernel_dx = kernel_dx.view(1, 1, 1, 3)
    kernel_dy = kernel_dy.view(1, 1, 3, 1)

    b, t, c, h, w = field.shape
    flat = field.view(b * t, c, h, w)

    dCdx = F.conv2d(flat, kernel_dx.expand(c, 1, 1, 3), padding=(0, 1), groups=c)
    dCdy = F.conv2d(flat, kernel_dy.expand(c, 1, 3, 1), padding=(1, 0), groups=c)

    # second derivatives (Laplacian components)
    kernel_dx2 = torch.tensor([[1, -2, 1]], dtype=field.dtype, device=field.device) / (dx ** 2)
    kernel_dy2 = kernel_dx2.transpose(0, 1)
    kernel_dx2 = kernel_dx2.view(1, 1, 1, 3)
    kernel_dy2 = kernel_dy2.view(1, 1, 3, 1)

    d2Cdx2 = F.conv2d(flat, kernel_dx2.expand(c, 1, 1, 3), padding=(0, 1), groups=c)
    d2Cdy2 = F.conv2d(flat, kernel_dy2.expand(c, 1, 3, 1), padding=(1, 0), groups=c)

    dCdx = dCdx.view(b, t, c, h, w)
    dCdy = dCdy.view(b, t, c, h, w)
    d2Cdx2 = d2Cdx2.view(b, t, c, h, w)
    d2Cdy2 = d2Cdy2.view(b, t, c, h, w)

    return dCdx, dCdy, d2Cdx2, d2Cdy2


def temporal_gradient(field, dt):
    """
    field: (B, T, C, H, W) -> dC/dt via forward difference along T
    Returns tensor with T-1 timesteps.
    """
    return (field[:, 1:] - field[:, :-1]) / dt


class PhysicsInformedLoss(nn.Module):
    def __init__(self, diffusion_coeff, dx, dy, dt, u=0.0, v=0.0):
        """
        u, v: advection velocity components (can be learned/estimated
              from data, or set to 0 for pure diffusion assumption).
        """
        super().__init__()
        self.D = diffusion_coeff
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.u = u
        self.v = v
        self.mse = nn.MSELoss()

    def physics_residual(self, pred_sequence):
        """
        pred_sequence: (B, T, C, H, W) — predicted frames, T >= 2
        Computes PDE residual: dC/dt + u*dC/dx + v*dC/dy - D*(d2C/dx2 + d2C/dy2)
        A perfect physical solution drives this residual to ~0.
        """
        dCdt = temporal_gradient(pred_sequence, self.dt)  # (B, T-1, C, H, W)

        dCdx, dCdy, d2Cdx2, d2Cdy2 = spatial_gradients(pred_sequence, self.dx, self.dy)
        # align spatial derivs to T-1 (drop last timestep to match dCdt)
        dCdx, dCdy = dCdx[:, :-1], dCdy[:, :-1]
        d2Cdx2, d2Cdy2 = d2Cdx2[:, :-1], d2Cdy2[:, :-1]

        residual = dCdt + self.u * dCdx + self.v * dCdy - self.D * (d2Cdx2 + d2Cdy2)
        return residual

    def forward(self, pred, target, lambda_physics):
        """
        pred, target: (B, T, C, H, W)
        """
        mse_loss = self.mse(pred, target)

        # Non-negativity penalty (e.g. rainfall/energy cannot be negative)
        negativity_penalty = torch.mean(torch.relu(-pred))

        if lambda_physics > 0:
            residual = self.physics_residual(pred)
            physics_loss = torch.mean(residual ** 2)
        else:
            physics_loss = torch.tensor(0.0, device=pred.device)

        total_loss = mse_loss + lambda_physics * (physics_loss + negativity_penalty)

        return total_loss, {
            "mse": mse_loss.item(),
            "physics": physics_loss.item() if torch.is_tensor(physics_loss) else physics_loss,
            "negativity_penalty": negativity_penalty.item(),
            "lambda": lambda_physics,
        }


class AdaptiveLambdaScheduler:
    """
    Data-first, physics-later curriculum:
      - epochs < warmup_epochs        -> lambda = lambda_init (~0, pure data fit)
      - warmup <= epoch < warmup+ramp -> lambda linearly ramps to lambda_max
      - epoch >= warmup+ramp          -> lambda = lambda_max
    """

    def __init__(self, lambda_init, lambda_max, warmup_epochs, ramp_epochs):
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

    def get_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return self.lambda_init
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            return self.lambda_init + progress * (self.lambda_max - self.lambda_init)
        else:
            return self.lambda_max