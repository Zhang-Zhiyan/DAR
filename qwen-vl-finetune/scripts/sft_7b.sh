#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# GPU 配置：使用哪些 GPU（按需修改）
export CUDA_VISIBLE_DEVICES=6,0
NPROC_PER_NODE=2  # 与 CUDA_VISIBLE_DEVICES 中的 GPU 数量一致

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=/path/to/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-5
batch_size=1
grad_accum_steps=16

# Keep single-sample video token count small enough for full fine-tuning.
max_pixels=12544
min_pixels=784
video_max_pixels=12544
video_min_pixels=784
video_max_frames=8
video_min_frames=2

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# 数据集名称（对应 data/__init__.py 中注册的名称）
datasets=dar_emotion_sft

# Output configuration
run_name="qwen2vl-baseline"
output_dir=/path/to/outputs/qwen2.5vl-3b-dar-sft-cr

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 4 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --video_max_pixels ${video_max_pixels} \
    --video_min_pixels ${video_min_pixels} \
    --video_max_frames ${video_max_frames} \
    --video_min_frames ${video_min_frames} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to none"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
