"""Inference script for the CameraBench evaluation set with CameraTransformer.

Same pipeline as inference.py, but reads CameraBench-style cases from a JSON file.
All paths are provided via command-line arguments.
"""

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from tqdm import tqdm

from vla import CameraTransformer
from utils.visualization import visualize_trajectory
from utils.pose_ex_enc import pose_encoding_to_extri
from utils.load_fn import load_and_preprocess_images
from utils.enhance_cam_pose import rescale_extrinsics_t_auto_numpy, densify_extrinsics
from utils.merge_results import stitch_images_horizontal

from prismatic.conf import ModelConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform

# Number of camera poses predicted per inference.
PREDICT_FRAME_NUM = 15

# Default pinhole intrinsics used for visualization.
DEFAULT_FX, DEFAULT_FY, DEFAULT_CX, DEFAULT_CY = 288.0, 512.0, 256.0, 256.0


def build_intrinsics(fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY):
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


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


def load_backbones(model_cfg, hf_token=None):
    """Load the vision and LLM backbones for the given base VLM config."""
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        model_cfg.image_resize_strategy,
    )
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=True,
    )
    print("### Vision_backbone and llm_backbone load finished.")
    return vision_backbone, image_transform, llm_backbone, tokenizer


def local_load_vla(
    model_id_or_path: Union[str, Path],
    model_cfg,
    vision_backbone,
    llm_backbone,
    norm_stats,
    load_for_training: bool = False,
    **kwargs,
) -> CameraTransformer:
    return CameraTransformer.from_pretrained(
        Path(model_id_or_path),
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        freeze_weights=not load_for_training,
        norm_stats=norm_stats,
        **kwargs,
    )


def model_estimation(model, image_pth, prompt, result_saved_pth, unnorm_key):
    image = Image.open(image_pth)
    actions, _ = model.predict_action(
        image,
        prompt,
        unnorm_key=unnorm_key,
        cfg_scale=1.5,
        use_ddim=True,
        num_ddim_steps=10,
    )
    print(actions.shape)
    torch.save(actions, result_saved_pth)
    return actions


def parse_args():
    parser = argparse.ArgumentParser(description="CameraTransformer CameraBench inference.")
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Experiment run directory containing config.json, "
                             "dataset_statistics.json and checkpoints/.")
    parser.add_argument("--cases-json", required=True, type=Path,
                        help="JSON file with a list of {image_pth, prompt} inference cases.")
    parser.add_argument("--output-root", default="outputs_camerabench")
    parser.add_argument("--gather-root", default="gather_results_camerabench")
    parser.add_argument("--checkpoint-filter", default="-epoch-05",
                        help="Substring used to select which checkpoints to run.")
    parser.add_argument("--unnorm-key", default="custom_finetuning")
    parser.add_argument("--action-model-type", default="DiT-B",
                        choices=["DiT-S", "DiT-B", "DiT-L"])
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_dir = args.run_dir
    config_json = run_dir / "config.json"
    dataset_statistics_json = run_dir / "dataset_statistics.json"
    assert config_json.exists(), f"Missing `config.json` for `{run_dir}`"
    assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir}`"

    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()

    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    vision_backbone, _, llm_backbone, _ = load_backbones(model_cfg, hf_token=args.hf_token)

    with open(args.cases_json, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    model_dir = run_dir / "checkpoints"
    checkpoints = sorted(os.listdir(model_dir))

    for checkpoint in tqdm(checkpoints):
        if args.checkpoint_filter not in checkpoint or "optimizer" in checkpoint:
            continue

        model_pth = model_dir / checkpoint
        model = local_load_vla(
            model_pth,
            model_cfg,
            vision_backbone,
            llm_backbone,
            norm_stats,
            load_for_training=False,
            action_model_type=args.action_model_type,
            future_action_window_size=PREDICT_FRAME_NUM,
        )
        model.to(device).eval()

        ckpt_tag = f"{run_dir.name}_{checkpoint.split('.')[0]}"

        for case in test_cases:
            image_pth = case["image_pth"]
            prompt = case["prompt"]
            print(f"Inference data: {image_pth} - {prompt}")

            test_example_name = os.path.basename(image_pth).replace(".png", "").split("@@")[-1]

            output_dir_pth = os.path.join(args.output_root, ckpt_tag)
            os.makedirs(output_dir_pth, exist_ok=True)

            output_save_root = os.path.join(output_dir_pth, test_example_name)
            os.makedirs(output_save_root, exist_ok=True)

            gather_results_save_root = os.path.join(args.gather_root, ckpt_tag)
            os.makedirs(gather_results_save_root, exist_ok=True)

            shutil.copy(image_pth, f"{output_save_root}/RefImage.jpg")
            with open(f"{output_save_root}/camera.txt", "a") as f:
                f.writelines(f"{prompt}\n")

            result_saved_pth = os.path.join(output_save_root, f"{checkpoint}_action.pt")
            pose_enc = model_estimation(model, image_pth, prompt, result_saved_pth, args.unnorm_key)

            pose_enc = torch.from_numpy(pose_enc).unsqueeze(0)
            extrinsic = pose_encoding_to_extri(pose_enc)
            extrinsic = extrinsic[:, :13, ...]

            extrinsic, _ = rescale_extrinsics_t_auto_numpy(extrinsic.squeeze(0).numpy())
            extrinsic = densify_extrinsics(extrinsic, inserts_per_gap=5)
            extrinsic = torch.from_numpy(extrinsic).unsqueeze(0)

            K = build_intrinsics()
            intrinsic = K.unsqueeze(0).repeat(extrinsic.shape[1], 1, 1)
            print(f"Extrinsic.shape: {extrinsic.shape}; Intrinsic.shape: {intrinsic.shape}")

            pred_txt_pth = os.path.join(output_save_root, f"{checkpoint}_action.txt")
            if os.path.exists(pred_txt_pth):
                open(pred_txt_pth, "w").close()
            convert_camera_pose_to_txt(intrinsic, extrinsic, pred_txt_pth)

            trajectory_visual_save_pth = os.path.join(output_save_root, f"{checkpoint}_action.png")
            visualize_trajectory(pred_txt_pth, trajectory_visual_save_pth, title="Pred Camera Trajectory")

            stitch_images_horizontal(
                [image_pth, trajectory_visual_save_pth],
                [" ", " "],
                prompt,
                out_path=f"{gather_results_save_root}/{test_example_name}.jpg",
            )

        del model
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        time.sleep(2)


if __name__ == "__main__":
    main()
