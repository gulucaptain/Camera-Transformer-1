"""
Wavelet validation for camera trajectories (SE(3) extrinsics)

Pipeline:
  extrinsics [B,N,3,4]  ->  T [B,N,4,4]
  -> relative motion ΔT_t = inv(T_{t-1}) @ T_t
  -> se(3) twist xi_t = log(ΔT_t)  # [B,N-1,6]
  -> Haar DWT along time on xi (per-dim, batched)
  -> low-only / high-only reconstruction in xi-space
  -> exp + cumulative compose to recover poses
  -> metrics: energy by scale, smoothness (acc/jerk), endpoint drift, rotation error

Dependencies: torch only.
"""

import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F

import pandas as pd

import matplotlib.pyplot as plt

# -----------------------------
# SO(3) / SE(3) utilities
# -----------------------------

def _skew(w: torch.Tensor) -> torch.Tensor:
    """w: [...,3] -> skew matrix [...,3,3]"""
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)
    K = torch.stack([
        torch.stack([O, -wz, wy], dim=-1),
        torch.stack([wz, O, -wx], dim=-1),
        torch.stack([-wy, wx, O], dim=-1),
    ], dim=-2)
    return K


def so3_exp(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Exponential map for SO(3).
    w: [...,3] axis-angle vector
    returns R: [...,3,3]
    """
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # [...,1]
    K = _skew(w)  # [...,3,3]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(K.shape)

    # Use series for small angles
    theta2 = theta * theta
    A = torch.where(theta > 1e-4, torch.sin(theta) / (theta + eps), 1 - theta2 / 6 + theta2 * theta2 / 120)
    B = torch.where(theta > 1e-4, (1 - torch.cos(theta)) / (theta2 + eps), 0.5 - theta2 / 24 + theta2 * theta2 / 720)

    A = A[..., None]  # [...,1,1]
    B = B[..., None]
    R = I + A * K + B * (K @ K)
    return R


def so3_log(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Log map for SO(3).
    R: [...,3,3]
    returns w: [...,3] axis-angle vector
    """
    # Clamp trace for numerical stability
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (tr - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # [...]

    # w_hat = (R - R^T) / (2 sin theta) * theta
    Rt = R.transpose(-1, -2)
    vee = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                       R[..., 0, 2] - R[..., 2, 0],
                       R[..., 1, 0] - R[..., 0, 1]], dim=-1)  # [...,3]

    sin_theta = torch.sin(theta)
    # For small angles: w ≈ 0.5 * vee
    small = theta < 1e-4
    scale = theta / (2 * (sin_theta + eps))
    scale = torch.where(small, torch.full_like(scale, 0.5), scale)
    w = scale[..., None] * vee
    return w


def se3_exp(xi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Exponential map for SE(3).
    xi: [...,6] where first 3 are omega, last 3 are v
    returns T: [...,4,4]
    """
    w = xi[..., :3]
    v = xi[..., 3:]
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # [...,1]
    W = _skew(w)
    R = so3_exp(w, eps=eps)
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(W.shape)

    theta2 = theta * theta
    theta3 = theta2 * theta

    A = torch.where(theta > 1e-4, (1 - torch.cos(theta)) / (theta2 + eps), 0.5 - theta2 / 24 + theta2 * theta2 / 720)
    B = torch.where(theta > 1e-4, (theta - torch.sin(theta)) / (theta3 + eps), 1/6 - theta2 / 120 + theta2 * theta2 / 5040)

    A = A[..., None]  # [...,1,1]
    B = B[..., None]
    V = I + A * W + B * (W @ W)  # [...,3,3]
    t = (V @ v[..., None]).squeeze(-1)  # [...,3]

    # Build homogeneous
    T = torch.zeros(*R.shape[:-2], 4, 4, device=xi.device, dtype=xi.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_log(T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Log map for SE(3).
    T: [...,4,4]
    returns xi: [...,6]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    w = so3_log(R, eps=eps)  # [...,3]
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # [...,1]
    W = _skew(w)
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand(W.shape)

    theta2 = theta * theta

    # V_inv approximation (Barfoot / standard)
    # V = I + (1-cosθ)/θ^2 W + (θ-sinθ)/θ^3 W^2
    # V^{-1} = I - 0.5 W + (1/θ^2) * (1 - (θ sinθ)/(2(1-cosθ))) W^2
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    # Handle small angles with series
    small = theta < 1e-4

    # c = (1/θ^2) * (1 - (θ sinθ)/(2(1-cosθ)))
    denom = 2 * (1 - cos_theta) + eps
    c = (1.0 / (theta2 + eps)) * (1 - (theta * sin_theta) / denom)

    # series c ≈ 1/12 + θ^2/720 ...
    c_series = (1/12) + theta2 / 720 + (theta2 * theta2) / 30240
    c = torch.where(small, c_series, c)

    V_inv = I - 0.5 * W + c[..., None] * (W @ W)
    v = (V_inv @ t[..., None]).squeeze(-1)
    xi = torch.cat([w, v], dim=-1)
    return xi


def invert_T(T: torch.Tensor) -> torch.Tensor:
    """Invert homogeneous transform: [...,4,4]"""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    tinv = -(Rt @ t[..., None]).squeeze(-1)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = tinv
    out[..., 3, 3] = 1.0
    return out


def extrinsics_to_T(extr: torch.Tensor) -> torch.Tensor:
    """extr: [B,N,3,4] -> T: [B,N,4,4]"""
    B, N = extr.shape[:2]
    T = torch.zeros(B, N, 4, 4, device=extr.device, dtype=extr.dtype)
    T[..., :3, :4] = extr
    T[..., 3, 3] = 1.0
    return T


def T_to_extrinsics(T: torch.Tensor) -> torch.Tensor:
    """T: [B,N,4,4] -> extr: [B,N,3,4]"""
    return T[..., :3, :4]


def relative_twist_from_extrinsics(extr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    extr: [B,N,3,4]
    returns:
      xi: [B,N-1,6] twists for ΔT_t = inv(T_{t-1})@T_t
      T0: [B,4,4] initial pose
    """
    T = extrinsics_to_T(extr)  # [B,N,4,4]
    T0 = T[:, 0]  # [B,4,4]
    T_prev = T[:, :-1]
    T_next = T[:, 1:]
    dT = invert_T(T_prev) @ T_next  # [B,N-1,4,4]
    xi = se3_log(dT)  # [B,N-1,6]
    return xi, T0


def integrate_twist(xi: torch.Tensor, T0: torch.Tensor) -> torch.Tensor:
    """
    xi: [B,T,6] (T = N-1)
    T0: [B,4,4]
    returns T: [B,T+1,4,4]
    """
    B, T = xi.shape[:2]
    Ts = torch.zeros(B, T + 1, 4, 4, device=xi.device, dtype=xi.dtype)
    Ts[:, 0] = T0
    cur = T0
    for i in range(T):
        cur = cur @ se3_exp(xi[:, i])
        Ts[:, i + 1] = cur
    return Ts


# -----------------------------
# Haar wavelet (DWT/IDWT) in torch
# -----------------------------

@dataclass
class HaarDWTResult:
    A: torch.Tensor              # [B, T_final, C]
    Ds: List[torch.Tensor]       # list of detail coeffs, each [B, T_l, C]
    orig_len: int
    pad_len: int


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
    """
    Reflect-pad along time to make length divisible by `multiple`.
    x: [B,T,C]
    returns (x_pad, pad_len)
    """
    B, T, C = x.shape
    r = T % multiple
    if r == 0:
        return x, 0
    pad_len = multiple - r
    # reflect pad on the right
    x_t = x.transpose(1, 2)  # [B,C,T]
    x_pad = F.pad(x_t, (0, pad_len), mode="reflect").transpose(1, 2)
    return x_pad, pad_len


def haar_dwt(x: torch.Tensor, levels: int) -> HaarDWTResult:
    """
    Multi-level Haar DWT along time.
    x: [B,T,C]
    Returns A_L and details D_1..D_L (D_1 highest frequency).
    """
    assert x.dim() == 3, "x must be [B,T,C]"
    B, T, C = x.shape
    multiple = 2 ** levels
    x_pad, pad_len = _pad_to_multiple(x, multiple)
    a = x_pad
    Ds: List[torch.Tensor] = []
    inv_sqrt2 = 1 / math.sqrt(2)

    for _ in range(levels):
        # pairwise
        even = a[:, 0::2, :]
        odd = a[:, 1::2, :]
        approx = (even + odd) * inv_sqrt2
        detail = (even - odd) * inv_sqrt2
        Ds.append(detail)  # D_1 first, then D_2, ...
        a = approx

    return HaarDWTResult(A=a, Ds=Ds, orig_len=T, pad_len=pad_len)


def haar_idwt(res: HaarDWTResult, use_low: bool = True, use_high: bool = True,
              keep_detail_levels: Optional[List[int]] = None) -> torch.Tensor:
    """
    Inverse Haar transform.
    - use_low: whether to use A_L
    - use_high: whether to use details Ds
    - keep_detail_levels: optional list of indices (1..L) to keep; others zeroed.
      D_1 is highest freq (first in res.Ds).
    Returns x_rec: [B,T,C] (cropped to original length)
    """
    A = res.A if use_low else torch.zeros_like(res.A)
    Ds = []
    L = len(res.Ds)
    for i, d in enumerate(res.Ds, start=1):
        if not use_high:
            Ds.append(torch.zeros_like(d))
        elif keep_detail_levels is not None and (i not in keep_detail_levels):
            Ds.append(torch.zeros_like(d))
        else:
            Ds.append(d)

    inv_sqrt2 = 1 / math.sqrt(2)
    a = A
    # reconstruct from level L down to 1
    for l in reversed(range(L)):
        d = Ds[l]
        even = (a + d) * inv_sqrt2
        odd = (a - d) * inv_sqrt2
        # interleave
        Tn = even.shape[1] + odd.shape[1]
        x = torch.empty(a.shape[0], Tn, a.shape[2], device=a.device, dtype=a.dtype)
        x[:, 0::2, :] = even
        x[:, 1::2, :] = odd
        a = x

    # crop padding
    x_rec = a[:, :res.orig_len, :]
    return x_rec


# -----------------------------
# Metrics / validation
# -----------------------------

def energy_by_scale(res: HaarDWTResult) -> Dict[str, torch.Tensor]:
    """
    Returns dict with:
      'E_low': [B,C] energy of A_L
      'E_D1'..: [B,C] energies of detail levels
      'E_total': [B,C]
      'ratio_low': [B,C]
      'ratio_Di': [B,C]
    """
    A = res.A  # [B,Tf,C]
    E_low = (A ** 2).sum(dim=1)  # [B,C]
    Es = []
    for d in res.Ds:
        Es.append((d ** 2).sum(dim=1))  # [B,C]
    E_total = E_low + torch.stack(Es, dim=0).sum(dim=0)
    out = {"E_low": E_low, "E_total": E_total, "ratio_low": E_low / (E_total + 1e-8)}
    for i, E in enumerate(Es, start=1):
        out[f"E_D{i}"] = E
        out[f"ratio_D{i}"] = E / (E_total + 1e-8)
    return out


def smoothness_metrics(xi: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    xi: [B,T,6]
    Returns:
      acc_energy: scalar per-batch [B]
      jerk_energy: [B]
    """
    B, T = xi.shape[:2]
    if T < 3:
        return {"acc_energy": xi.new_zeros(B), "jerk_energy": xi.new_zeros(B)}
    dd = xi[:, 2:, :] - 2 * xi[:, 1:-1, :] + xi[:, :-2, :]
    acc = (dd ** 2).mean(dim=(1, 2))
    if T < 4:
        return {"acc_energy": acc, "jerk_energy": xi.new_zeros(B)}
    ddd = dd[:, 1:, :] - dd[:, :-1, :]
    jerk = (ddd ** 2).mean(dim=(1, 2))
    return {"acc_energy": acc, "jerk_energy": jerk}


def rotation_geodesic(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """R1,R2: [...,3,3] -> angle [...], in radians"""
    R = R1.transpose(-1, -2) @ R2
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (tr - 1) / 2
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.acos(cos)


def pose_error(T_gt: torch.Tensor, T_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    T_gt, T_pred: [B,N,4,4]
    Returns:
      trans_rmse: [B]
      rot_mean: [B] (radians)
      endpoint_trans: [B] (norm of final translation error)
      endpoint_rot: [B] (radians)
    """
    t_gt = T_gt[..., :3, 3]
    t_pr = T_pred[..., :3, 3]
    trans_rmse = torch.sqrt(((t_gt - t_pr) ** 2).mean(dim=(1, 2)))
    # mean rotation error per frame
    R_gt = T_gt[..., :3, :3]
    R_pr = T_pred[..., :3, :3]
    ang = rotation_geodesic(R_gt.reshape(-1, 3, 3), R_pr.reshape(-1, 3, 3)).reshape(T_gt.shape[0], -1)
    rot_mean = ang.mean(dim=1)

    endpoint_trans = torch.linalg.norm(t_gt[:, -1] - t_pr[:, -1], dim=-1)
    endpoint_rot = rotation_geodesic(R_gt[:, -1], R_pr[:, -1])
    return {
        "trans_rmse": trans_rmse,
        "rot_mean": rot_mean,
        "endpoint_trans": endpoint_trans,
        "endpoint_rot": endpoint_rot,
    }


# -----------------------------
# Main validation function
# -----------------------------

@torch.no_grad()
def wavelet_validate_extrinsics(
    extr: torch.Tensor,
    levels: int = 3,
    keep_detail_levels: Optional[List[int]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    extr: [B,N,3,4] camera extrinsics, assumed to be valid rigid transforms.
    levels: Haar DWT levels on the motion sequence length (N-1).
    keep_detail_levels: optional list like [1,2] to keep only highest freq details.

    Returns a dict with sections:
      energy, smoothness, errors
    """
    assert extr.dim() == 4 and extr.shape[-2:] == (3, 4), "extr must be [B,N,3,4]"
    B, N = extr.shape[:2]
    assert N >= 4, "Need at least 4 frames for meaningful smoothness metrics."

    xi, T0 = relative_twist_from_extrinsics(extr)  # [B,N-1,6], [B,4,4]
    T_gt = extrinsics_to_T(extr)

    # DWT on xi along time
    res = haar_dwt(xi, levels=levels)
    energy = energy_by_scale(res)

    # Reconstruct low-only and high-only
    xi_low = haar_idwt(res, use_low=True, use_high=False)
    xi_high = haar_idwt(res, use_low=False, use_high=True, keep_detail_levels=keep_detail_levels)

    # Integrate back to poses
    T_low = integrate_twist(xi_low, T0)
    T_high = integrate_twist(xi_high, T0)

    # Smoothness metrics on motion
    sm_gt = smoothness_metrics(xi)
    sm_low = smoothness_metrics(xi_low)
    sm_high = smoothness_metrics(xi_high)

    # Pose errors vs GT
    err_low = pose_error(T_gt, T_low)
    err_high = pose_error(T_gt, T_high)

    return {
        "energy": energy,
        "smoothness": {
            "gt": sm_gt,
            "low_only": sm_low,
            "high_only": sm_high,
        },
        "errors": {
            "low_only": err_low,
            "high_only": err_high,
        },
        "debug_shapes": {
            "xi": torch.tensor(list(xi.shape), device=extr.device),
            "A": torch.tensor(list(res.A.shape), device=extr.device),
            "D1": torch.tensor(list(res.Ds[0].shape), device=extr.device),
        }
    }


def _make_dummy_extrinsics(B=2, N=33, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Create a smooth camera trajectory with small jitter for demo.
    Output: [B,N,3,4] camera-to-world like transforms (still rigid).
    """
    t = torch.linspace(0, 1, N, device=device, dtype=dtype)
    # smooth orbit-ish translation
    tx = 0.5 * torch.cos(2 * math.pi * t)
    ty = 0.2 * torch.sin(2 * math.pi * t)
    tz = 0.3 * t
    trans = torch.stack([tx, ty, tz], dim=-1)  # [N,3]

    # smooth yaw rotation
    yaw = 0.6 * torch.sin(2 * math.pi * t)  # [N]
    w = torch.stack([torch.zeros_like(yaw), yaw, torch.zeros_like(yaw)], dim=-1)  # [N,3]
    R = so3_exp(w)  # [N,3,3]

    # add small high-freq jitter in motion space by perturbing per-frame rotation slightly
    jitter = 0.02 * torch.randn(N, 3, device=device, dtype=dtype)
    R = R @ so3_exp(jitter)

    extr = torch.zeros(B, N, 3, 4, device=device, dtype=dtype)
    extr[:, :, :3, :3] = R[None].expand(B, -1, -1, -1)
    extr[:, :, :3, 3] = trans[None].expand(B, -1, -1)
    return extr
