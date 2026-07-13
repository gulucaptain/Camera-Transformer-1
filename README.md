<div align="center">


<img src="assets/logo.png" alt="CT-1 Logo" width="500" />

<h2>
  CT-1: Vision-Language-Camera Models Transfer Spatial Reasoning Knowledge to Camera-controllable Video Generation
</h2>

<p>
  <a href="https://gulucaptain.github.io/Camera-Transformer-1/"><img src="https://img.shields.io/badge/🌐_Project_Page-CT--1-blue?style=for-the-badge" alt="Project Page"/></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2604.09201"><img src="https://img.shields.io/badge/📄_Paper-ArXiv-red?style=for-the-badge" alt="ArXiv"/></a>
  &nbsp;
  <a href="https://github.com/gulucaptain/Camera-Transformer-1"><img src="https://img.shields.io/badge/💻_Code-Available-green?style=for-the-badge" alt="Code"/></a>
</p>

<p>
  <strong>Haoyu Zhao, Zihao Zhang, Jiaxi Gu, Haoran Chen, Qingping Zheng, Pin Tang, Yeyin Jin, <br> Yuang Zhang, Junqi Cheng, Zenghui Lu, Peng Shu, Zuxuan Wu, Yu-Gang Jiang<br></strong>
  <em>Fudan University; Tencent.</em>
</p>

<p>
  <img src="https://img.shields.io/github/stars/gulucaptain/Camera-Transformer-1?style=social" alt="Stars"/>
</p>

</div>

---

## 📋 Abstract

Camera-controllable video generation aims to synthesize videos with flexible and physically plausible camera movements. However, existing methods either provide imprecise camera control from text prompts or rely on labor-intensive manual camera trajectory parameters, limiting their use in automated scenarios.

To address these issues, we propose a novel **Vision-Language-Camera model**, termed **CT-1 (Camera Transformer 1)**, a specialized model designed to *transfer spatial reasoning knowledge to video generation* by accurately estimating camera trajectories. Built upon vision-language modules and a Diffusion Transformer model, CT-1 employs a *Wavelet-based Regularization Loss* in the frequency domain to effectively learn complex camera trajectory distributions. These trajectories are integrated into a video diffusion model to enable spatially aware camera control that aligns with user intentions.

To facilitate the training of CT-1, we design a dedicated data curation pipeline and construct **CT-200K**, a large-scale dataset containing over *47M frames*. Experimental results demonstrate that our framework successfully bridges the gap between spatial reasoning and video synthesis, yielding faithful and high-quality camera-controllable videos and improving camera control accuracy by 25.7% over prior methods.

---

## 🔥 News

| Date | Event |
|------|-------|
| 🟢 2026-07-14 | Code and training/inference pipeline released. |
| 🟡 2026-04-10 | Project page released. |

---

## 🎬 Video Generation with CT-1

> **Challenging Scenarios** — Forward motion & rotational motion across diverse scenes.
>
> 🔁 *Animated previews below (GIF).*

<table>
  <tr>
    <td align="center"><img src="assets/ct_1_shows/show_case_1.gif" width="768"/></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/ct_1_shows/show_case_2.gif" width="768"/></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/ct_1_shows/show_case_3.gif" width="768"/></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/ct_1_shows/show_case_4.gif" width="768"/></td>
  </tr>
</table>

> 💡 For full video demos including camera trajectory visualizations, cross-model comparisons, and driving scenarios, please visit our [**Project Page**](https://gulucaptain.github.io/Camera-Transformer-1/).

---

## 🧠 Framework Overview

CT-1 follows a **"Camera-Decision-First, Generation-Next"** two-stage paradigm:

```
Vision-Language Input (Image + Text)
          │
          ▼
  ┌───────────────────┐
  │  CT-1 (VLC Model) │  ← Diffusion Transformer + Wavelet Regularization Loss
  └───────────────────┘
          │
          ▼
   Camera Trajectories
          │
          ▼
  ┌─────────────────────────┐
  │  Video Diffusion Model  │  ← Camera controllable video generation
  └─────────────────────────┘
          │
          ▼
   Generated Video
```

The framework consists of three main components:
- **(a) Vision-Language Module** — for semantic embedding of image and text inputs
- **(b) Diffusion Transformer Module** — for modeling camera trajectory distributions with Wavelet-based Regularization Loss
- **(c) Controllable Video Generation Models** — synthesize videos conditioned on the predicted trajectories

This repository contains the code for **stage (a)+(b)**: the CT-1 Vision-Language-Camera
model that estimates camera trajectories from an image and a text prompt. The predicted
trajectories are compatible with existing camera-controllable video diffusion models
(e.g., CameraCtrl, MotionCtrl).

---

## ✨ Highlights

- 🎯 **VLC Model**: First to formulate camera trajectory estimation as a vision-language understanding task
- 🌊 **Wavelet-based Regularization Loss**: Novel frequency-domain loss for learning complex camera trajectory distributions
- 📦 **CT-200K Dataset**: Large-scale dataset with 47M+ frames and dedicated curation pipeline
- 🔌 **Cross-Model Compatibility**: CT-1 predicted trajectories are compatible with existing models (CameraCtrl, MotionCtrl, etc.)
- 🚗 **Cross-Domain Generalization**: Validated on general scenes and driving scenarios

---

## 📑 Contents
 * [**Installation**](#-installation)
 * [**Getting Started**](#-getting-started)
 * [**Training**](#-training)
 * [**Inference**](#-inference)
 * [**Deployment**](#-deployment)
 * [**Video Generation with CT-1**](#-video-generation-with-ct-1)
 * [**Citation**](#-citation)
 * [**License**](#-license)

---

## 🛠 Installation

The code is built using Python 3.10, and can be run under any environment with Python 3.8
and above. We require PyTorch >= 2.2.0 and CUDA >= 12.0 (it may run with lower versions,
but this has not been tested).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html):
```bash
conda create --name camera_transformer python=3.10
conda activate camera_transformer
```
Clone the repository and install the required packages:
```bash
git clone https://github.com/gulucaptain/Camera-Transformer-1
cd Camera-Transformer-1
pip install -e .
```
If you need the training code, please also install
[Flash Attention](https://github.com/Dao-AILab/flash-attention):
```bash
pip install -e .[train]
```
or install it manually:
```bash
# Training additionally requires Flash-Attention 2
pip install packaging ninja

# Verify Ninja --> should return exit code "0"
ninja --version; echo $?

# Install Flash Attention 2
# =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install "flash-attn==2.5.5" --no-build-isolation
```

---

## 🚀 Getting Started

Load a trained CT-1 checkpoint and run a minimal inference. The model takes an image and a
text prompt describing the desired camera motion, and returns a predicted **camera
trajectory** (a sequence of camera-pose encodings):

```python
from PIL import Image
from vla import load_vla
import torch

model = load_vla(
        '/path/to/camera-transformer-checkpoint',  # local checkpoint directory or hub id
        load_for_training=False,
        action_model_type='DiT-B',                  # must match the trained model size ('DiT-S', 'DiT-B', 'DiT-L')
        future_action_window_size=15,
    )
# about 30G Memory in fp32

# (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load the vlm in bf16

model.to('cuda:0').eval()

image: Image.Image = <input_your_image>
prompt = "<describe the camera motion, e.g. 'the camera moves forward and rotates left'>"

# Predict the camera trajectory (a sequence of camera-pose encodings).
trajectory, _ = model.predict_action(
            image,
            prompt,
            unnorm_key='custom_finetuning',  # the de-normalization key of your dataset
            cfg_scale=1.5,                   # cfg from 1.5 to 7 also performs well
            use_ddim=True,                   # use DDIM sampling
            num_ddim_steps=10,               # number of steps for DDIM sampling
        )

# `trajectory` is a sequence of camera-pose encodings (one per predicted step),
# ready to be fed into a camera-controllable video diffusion model.
```

A batched variant, `predict_action_batch`, is available in
[vla/camera_transformer.py](./vla/camera_transformer.py) to accelerate inference.

> 📌 Trajectory visualization code is available in our companion repository:
> [Camera Trajectories Visualization](https://github.com/gulucaptain/Camera-Trajectories-Visualization).

---

## 🏋 Training

CT-1 is trained with PyTorch Fully Sharded Data Parallel
([FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)). The
vision-language backbone follows [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms),
and the Diffusion Transformer head is trained with our Wavelet-based Regularization Loss.

### Data format

Training data is expected in [RLDS](https://github.com/kpertsch/rlds_dataset_builder)
format. Convert your camera-trajectory dataset (e.g., CT-200K) to RLDS, place it under
`<data_root_dir>/custom_finetuning/1.0.0`, and set `--vla.data_mix custom_finetuning`.

### Fine-tuning from a pretrained checkpoint

If your checkpoints live on a private hub, create a
[Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens) and
export it (the training script reads it from the environment):
```bash
export HF_TOKEN=<your_hf_token>
```

Launch training (example: one node with 8 GPUs). See [`finetune.sh`](./finetune.sh) for a
ready-to-edit launch script:
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --pretrained_checkpoint /path/to/camera-transformer-checkpoint/checkpoints/camera-transformer-checkpoint.pt \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix custom_finetuning \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 32 \
  --vla.learning_rate 2e-5 \
  --data_root_dir /path/to/tensorflow_datasets \
  --run_root_dir /path/to/experiments \
  --run_id my_finetune_run \
  --image_aug False \
  --wandb_project camera_transformer_finetune \
  --wandb_entity your_wandb_entity \
  --save_interval 5000 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False
```
More training settings can be customized in [`conf/vla.py`](conf/vla.py) by registering a
new VLA type. To resume from a checkpoint, set `--is_resume True` together with
`--resume_step` and `--resume_epoch` matching the checkpoint.

### Training from scratch

For greater efficiency you can initialize the vision-language backbone from a
Prismatic / [OpenVLA](https://github.com/openvla/openvla) checkpoint rather than training it
from scratch. Download the OpenVLA weights following their instructions, then pass them via
`--pretrained_checkpoint`:
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --pretrained_checkpoint /path/to/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix custom_finetuning \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 32 \
  --vla.learning_rate 2e-5 \
  --data_root_dir /path/to/tensorflow_datasets \
  --run_root_dir /path/to/experiments \
  --run_id my_scratch_run \
  --image_aug False \
  --wandb_project camera_transformer_pretrain \
  --wandb_entity your_wandb_entity \
  --save_interval 5000 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False
```
You can also omit `--pretrained_checkpoint` to start from PrismaticVLM directly, though it
will take longer to converge.

---

## 🔎 Inference

Batch inference over a set of prompts is provided by
[`scripts/inference.py`](./scripts/inference.py). Cases are read from a JSON file
(see [`assets/inference_cases/example_prompts.json`](./assets/inference_cases/example_prompts.json)
for the expected `{image_pth, prompt}` format). Edit and run [`inference.sh`](./inference.sh):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
  --run-dir /path/to/experiments/my_finetune_run \
  --cases-json assets/inference_cases/example_prompts.json \
  --output-root outputs \
  --gather-root gather_results \
  --action-model-type DiT-B
```
For evaluation on the CameraBench-style split, use
[`scripts/camerabench_inference.py`](./scripts/camerabench_inference.py), which follows the
same pipeline but reads CameraBench-format cases.

---

## 🌐 Deployment

A minimal HTTP server/client for online trajectory prediction is provided in
[`scripts/deploy.py`](./scripts/deploy.py). Start the server pointing at your trained model:
```bash
python scripts/deploy.py \
  --saved_model_path /path/to/camera-transformer-checkpoint \
  --unnorm_key custom_finetuning \
  --use_bf16 \
  --cfg_scale 1.5 \
  --port 5500
```
The client only needs a Python environment with the `requests` library. Example request to
a server running on `127.0.0.1:5500`:
```python
import requests
import json

url = 'http://127.0.0.1:5500/api/inference'
data = {'task_description': "<describe the camera motion>"}
image = "image/example.png"

json.dump(data, open("data.json", "w"))

with open("data.json", "r") as query_file:
    with open(image, "rb") as image_file:
        files = [
            ('images', (image, image_file, 'image/png')),
            ('json', ("data.json", query_file, 'application/json')),
        ]
        response = requests.post(url, files=files)

if response.status_code != 200:
    print("Failed to get a response from the API")
    print(response.text)
```


---

## 📎 Citation

If you find this work useful, please consider citing:

```bibtex
@article{zhao2026ct1,
  title     = {CT-1: Vision-Language-Camera Models Transfer Spatial Reasoning Knowledge to Camera-controllable Video Generation},
  author    = {Haoyu Zhao, Zihao Zhang, Jiaxi Gu, Haoran Chen, Qingping Zheng, Pin Tang, Yeyin Jin, Yuang Zhang, Junqi Cheng, Zenghui Lu, Peng Shu, Zuxuan Wu, Yu-Gang Jiang},
  journal   = {The 34th ACM International Conference on Multimedia (ACM MM)},
  year      = {2026}
}
```

---

## 📄 License

All code, model weights, and data are licensed under the [MIT license](./LICENSE).

---

<a href="https://www.star-history.com/#gulucaptain/Camera-Transformer-1&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=gulucaptain/Camera-Transformer-1&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=gulucaptain/Camera-Transformer-1&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=gulucaptain/Camera-Transformer-1&type=Date" width="450" />
  </picture>
</a>

---

<div align="center">
  <sub>Built with ❤️ | <a href="https://gulucaptain.github.io/Camera-Transformer-1/">Project Page</a></sub>
</div>
