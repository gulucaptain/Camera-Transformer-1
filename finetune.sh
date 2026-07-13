# Example fine-tuning launch script.
#
# Fill in the placeholder paths (/path/to/...) with your own locations before running.
# For multi-node training, launch this script on each node with the appropriate
# --nnodes / --node_rank / --master_addr arguments for your cluster.

# Provide your own HuggingFace token via the environment, e.g.:
#   export HF_TOKEN=<your_hf_token>
# The training script reads it from os.environ.

export NCCL_DEBUG=INFO
export NCCL_ALGO=Ring

torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --pretrained_checkpoint "/path/to/camera-transformer-checkpoint/checkpoints/camera-transformer-checkpoint.pt" \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix "custom_finetuning" \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 128 \
  --vla.per_device_batch_size 16 \
  --vla.learning_rate 2e-5 \
  --vla.epochs 50 \
  --vla.weight_decay 0.01 \
  --vla.lr_scheduler_type linear-warmup+cosine-decay \
  --vla.warmup_ratio 0.1 \
  --vla.max_grad_norm 1.0 \
  --data_root_dir "/path/to/tensorflow_datasets" \
  --run_root_dir "/path/to/experiments" \
  --run_id "my_finetune_run" \
  --image_aug False \
  --wandb_project "camera_transformer_finetune" \
  --wandb_entity "your_wandb_entity" \
  --save_interval 5000 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False \
  --load_all_data_for_training True
