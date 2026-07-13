import os
import torch
import torch.nn.functional as F
from utils.pose_ex_enc import extri_to_pose_encoding, pose_encoding_to_extri
from tqdm import tqdm

from utils.visualization import visualize_trajectory

def smooth_translation(translation, window_size=5):
    """
    translation: [T,3]
    """
    T, C = translation.shape
    pad = window_size // 2
    smoothed = torch.zeros_like(translation)

    kernel = torch.ones(1,1,window_size, device=translation.device)/window_size

    for c in range(C):
        seq = translation[:,c].unsqueeze(0).unsqueeze(0)  # [1,1,T]
        seq_padded = F.pad(seq, (pad,pad), mode='replicate')
        smoothed_seq = F.conv1d(seq_padded, kernel, padding=0)  # [1,1,T]
        smoothed[:,c] = smoothed_seq[0,0]

    return smoothed, pad

def smooth_camera_extrinsics_3x4(extrinsics, window_size=5):
    """
    extrinsics: [T,3,4]  # 3x3 rotation (compressed) + 1x3 translation
    """
    T, C, D = extrinsics.shape
    assert D == 4

    # 平滑平移
    translation = extrinsics[:,:,3]  # [T,3]
    smoothed_translation, pad = smooth_translation(translation, window_size=5)

    # 平滑旋转
    rotation = extrinsics[:,:,:3]  # [T,3,3]
    rotation_smooth = torch.zeros_like(rotation)

    for t in range(T):
        start = max(0, t-pad)
        end = min(T, t+pad+1)
        R_window = rotation[start:end]  # [w,3,3]
        R_mean = R_window.mean(dim=0)  # 均值
        # SVD 重正交化
        U, _, V = torch.svd(R_mean)
        R_orth = U @ V.T
        rotation_smooth[t] = R_orth

    # 合并旋转和平移
    smoothed_extrinsics = torch.zeros_like(extrinsics)
    smoothed_extrinsics[:,:,:3] = rotation_smooth
    smoothed_extrinsics[:,:,3] = smoothed_translation

    return smoothed_extrinsics

def convert_camera_pose_to_txt(intrinsics, extrinsics, txt_pth):
    intrinsics = intrinsics.squeeze(0)
    extrinsics = extrinsics.squeeze(0)
    frame_nums = extrinsics.shape[0]

    with open(txt_pth, "a") as f:
        for i in tqdm(range(frame_nums)):
            intrinsic = intrinsics[i]
            extrinsic = extrinsics[i]
            intrinsic_value = f"{i} {intrinsic[0][0]} {intrinsic[1][1]} {intrinsic[0][2]} {intrinsic[1][2]} {0.000000000} {0.000000000}"
            extrinsic_value = ""
            for j in range(3):
                for k in range(4):
                    extrinsic_value += f"{extrinsic[j][k]} "
            value = f"{intrinsic_value} {extrinsic_value}"
            f.writelines(f"{value}\n")

def visualization(extrinsic_gt):
    txt_pth = "./action_fake_intrinsic.txt"
    with open(txt_pth, "w") as f:
        f.write("")
    
    frame_num = extrinsic_gt.shape[0]

    K = torch.tensor([
        [900.0,   0.0, 512.0],
        [  0.0, 900.0, 288.0],
        [  0.0,   0.0,   1.0]
    ])

    # 扩展到 n 帧
    intrinsic = K.unsqueeze(0).repeat(frame_num, 1, 1)
    print(f"intrinsic.shape: {intrinsic.shape}")
    print(f"extrinsic_gt.shape: {extrinsic_gt.shape}")
    convert_camera_pose_to_txt(intrinsic, extrinsic_gt, txt_pth)
    trajectory_visual_save_pth = "./action_fake7.png"
    visualize_trajectory(txt_pth, trajectory_visual_save_pth)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smooth camera extrinsics (translation + rotation) for a directory of trajectories."
    )
    parser.add_argument("--camera-pose-root", required=True,
                        help="Directory of per-instance folders each containing extrinsic.pt.")
    parser.add_argument("--output-root", required=True,
                        help="Directory to write the smoothed extrinsics to.")
    parser.add_argument("--window-size", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    camera_instances = sorted(os.listdir(args.camera_pose_root))
    for camera_instance in tqdm(camera_instances):
        extrinsic_pth = os.path.join(args.camera_pose_root, camera_instance, "extrinsic.pt")
        if os.path.exists(extrinsic_pth):
            extrinsic_gt = torch.load(extrinsic_pth, map_location=device)
            extrinsic_gt = extrinsic_gt.squeeze(0)

            extrinsic_gt = smooth_camera_extrinsics_3x4(extrinsic_gt, window_size=args.window_size)
            extrinsic_gt = torch.round(extrinsic_gt * 1000) / 1000
            extrinsic_gt = extrinsic_gt.unsqueeze(0)

            saved_dir = os.path.join(args.output_root, camera_instance)
            os.makedirs(saved_dir, exist_ok=True)
            torch.save(extrinsic_gt, os.path.join(saved_dir, "extrinsic.pt"))
