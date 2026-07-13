import numpy as np

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl

from tqdm import tqdm


def rescale_extrinsics_t_numpy(E: np.ndarray,
                               scale: float = 20.0,
                               keep_first_center: bool = True) -> np.ndarray:
    """
    E: [T, 3, 4]  each is [R|t]
    Only scales translation t. R stays unchanged.
    If keep_first_center=True, amplify relative motion w.r.t. first frame
    so the starting camera center stays fixed (good for visualization).
    """
    assert E.ndim == 3 and E.shape[1:] == (3, 4), f"Expected [T,3,4], got {E.shape}"
    E_out = E.copy()

    R = E_out[:, :, :3]      # [T,3,3]
    t = E_out[:, :, 3]       # [T,3]
    
    if keep_first_center:
        # Work in camera centers C in world coordinates:
        # For x_cam = R x_world + t, camera center is C = -R^T t
        C = -np.einsum('tji,tj->ti', R, t)   # [T,3]  R^T @ t with batch
        C0 = C[0:1]                           # [1,3]
        C_scaled = C0 + scale * (C - C0)      # amplify trajectory around the first center

        # Convert back: t = -R C
        t_scaled = -np.einsum('tij,tj->ti', R, C_scaled)
    else:
        # Simple: directly scale t (OK for pure viz, but origin may drift depending on convention)
        t_scaled = t * scale

    E_out[:, :, 3] = t_scaled
    return E_out


def rescale_extrinsics_t_auto_numpy(E: np.ndarray,
                                    target_avg_center_step: float = 0.5,
                                    keep_first_center: bool = True):
    assert E.ndim == 3 and E.shape[1:] == (3, 4), f"Expected [T,3,4], got {E.shape}"
    R = E[:, :, :3]
    t = E[:, :, 3]
    C = -np.einsum('tji,tj->ti', R, t)          # [T,3]
    steps = np.linalg.norm(C[1:] - C[:-1], axis=1)  # [T-1]
    mean_step = float(np.mean(steps)) + 1e-12

    scale = target_avg_center_step / mean_step * 2
    E_scaled = rescale_extrinsics_t_numpy(E, scale=scale, keep_first_center=keep_first_center)
    return E_scaled, scale


def _normalize_quat(q):
    return q / (np.linalg.norm(q) + 1e-12)

def _rotmat_to_quat(R):
    """
    R: [3,3] rotation matrix
    returns quaternion [w, x, y, z] with w>=0
    """
    m = R
    tr = np.trace(m)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2,1] - m[1,2]) / S
        y = (m[0,2] - m[2,0]) / S
        z = (m[1,0] - m[0,1]) / S
    else:
        if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
            w = (m[2,1] - m[1,2]) / S
            x = 0.25 * S
            y = (m[0,1] + m[1,0]) / S
            z = (m[0,2] + m[2,0]) / S
        elif m[1,1] > m[2,2]:
            S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
            w = (m[0,2] - m[2,0]) / S
            x = (m[0,1] + m[1,0]) / S
            y = 0.25 * S
            z = (m[1,2] + m[2,1]) / S
        else:
            S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
            w = (m[1,0] - m[0,1]) / S
            x = (m[0,2] + m[2,0]) / S
            y = (m[1,2] + m[2,1]) / S
            z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    # fix sign for continuity preference
    if q[0] < 0:
        q = -q
    return _normalize_quat(q)

def _quat_to_rotmat(q):
    """
    q: [w,x,y,z]
    """
    q = _normalize_quat(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R

def _slerp(q0, q1, u):
    """
    Spherical linear interpolation between q0 and q1 at u in [0,1]
    """
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)

    dot = float(np.dot(q0, q1))
    # take shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    # if very close, fall back to lerp
    if dot > 0.9995:
        q = q0 + u * (q1 - q0)
        return _normalize_quat(q)

    theta0 = np.arccos(dot)
    sin_theta0 = np.sin(theta0)
    theta = theta0 * u
    sin_theta = np.sin(theta)

    s0 = np.sin(theta0 - theta) / sin_theta0
    s1 = sin_theta / sin_theta0
    return _normalize_quat(s0 * q0 + s1 * q1)

def densify_extrinsics(E, inserts_per_gap=3):
    """
    E: [T,3,4]
    inserts_per_gap: number of NEW frames inserted between i and i+1
      - 0 => unchanged
      - 3 => each gap becomes 5 frames total (i + 3 new + i+1)
    return: [T',3,4]
    """
    assert E.ndim == 3 and E.shape[1:] == (3,4), f"Expected [T,3,4], got {E.shape}"
    T = E.shape[0]

    R_all = E[:, :, :3].astype(np.float64)
    t_all = E[:, :, 3].astype(np.float64)

    out = []
    for i in range(T - 1):
        R0, R1 = R_all[i], R_all[i+1]
        t0, t1 = t_all[i], t_all[i+1]
        q0, q1 = _rotmat_to_quat(R0), _rotmat_to_quat(R1)

        # add the start frame
        out.append(E[i])

        # insert intermediate frames
        for k in range(1, inserts_per_gap + 1):
            u = k / (inserts_per_gap + 1.0)
            qi = _slerp(q0, q1, u)
            Ri = _quat_to_rotmat(qi)
            ti = (1 - u) * t0 + u * t1
            Ei = np.concatenate([Ri, ti[:, None]], axis=1).astype(E.dtype)
            out.append(Ei)

    # add final frame
    out.append(E[-1])

    return np.stack(out, axis=0)
