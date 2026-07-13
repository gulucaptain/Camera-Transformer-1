import torch
from .rotation import quat_to_mat, mat_to_quat

def extri_to_pose_encoding(
    extrinsics, pose_encoding_type="absT_quaR_FoV"  # e.g., (256, 512)
):
    # extrinsics: BxSx3x4
    if pose_encoding_type == "absT_quaR_FoV":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3
        
        quat = mat_to_quat(R)
        
        pose_encoding = torch.cat([T, quat], dim=-1).float()
    else:
        raise NotImplementedError

    return pose_encoding

def pose_encoding_to_extri(
    pose_encoding, pose_encoding_type="absT_quaR_FoV",
):
    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)
    else:
        raise NotImplementedError

    return extrinsics
