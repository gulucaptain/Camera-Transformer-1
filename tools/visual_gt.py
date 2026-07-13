from PIL import Image
from vla import load_vla
import torch
import os
from tqdm import tqdm

from utils.load_fn import load_and_preprocess_images
from utils.visualization import visualize_trajectory

from utils.pose_enc import pose_encoding_to_extri_intri
from utils.pose_ex_enc import pose_encoding_to_extri

import gc
import time

PREDICT_FRAME_NUM = 15


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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize ground-truth camera trajectories at varying temporal strides."
    )
    parser.add_argument("--image-path", required=True, help="Reference image path.")
    parser.add_argument("--camera-pose-dir", required=True,
                        help="Directory of per-instance folders with intrinsic.pt / extrinsic.pt.")
    parser.add_argument("--output-root", default="./outputs_gt")
    parser.add_argument("--pose-prefix", default="",
                        help="Prefix prepended to the image stem to locate the pose folder.")
    args = parser.parse_args()

    image_pth = args.image_path
    test_example_name = os.path.basename(image_pth).split(".")[0].replace("-", "")
    output_save_root = os.path.join(args.output_root, test_example_name)
    os.makedirs(output_save_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for index in range(7, 20):
        txt_pth = os.path.join(output_save_root, "action_gt.txt")
        if os.path.exists(txt_pth):
            open(txt_pth, "w").close()

        exist_file_length = len(os.listdir(output_save_root))
        gt_trajectory_visual_save_pth = os.path.join(output_save_root, f"action_gt_{exist_file_length + 1}.png")

        if not os.path.exists(gt_trajectory_visual_save_pth):
            image_stem = os.path.basename(image_pth).split(".")[0]
            camera_pose_pth = os.path.join(args.camera_pose_dir, f"{args.pose_prefix}{image_stem}")
            intrinsic_pth = os.path.join(camera_pose_pth, "intrinsic.pt")
            extrinsic_pth = os.path.join(camera_pose_pth, "extrinsic.pt")
            intrinsic_gt = torch.load(intrinsic_pth, map_location=device).squeeze(0)
            extrinsic_gt = torch.load(extrinsic_pth, map_location=device).squeeze(0)
            frame_num = extrinsic_gt.shape[0]

            num_frames = PREDICT_FRAME_NUM
            stride = index
            indices = torch.arange(0, frame_num, stride)[:num_frames]
            intrinsic_gt = intrinsic_gt[indices]
            extrinsic_gt = extrinsic_gt[indices]

            print(f"intrinsic_gt.shape: {intrinsic_gt.shape}")
            print(f"extrinsic_gt.shape: {extrinsic_gt.shape}")

            convert_camera_pose_to_txt(intrinsic_gt, extrinsic_gt, txt_pth)
            visualize_trajectory(txt_pth, gt_trajectory_visual_save_pth, title="GT Camera Trajectory")
