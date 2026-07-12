#!/bin/bash
# =============================================================================
# Qwen3-VL-8B-Instruct SFT 训练脚本
# 任务：Video Emotion Segmentation & Reasoning
# 硬件：8 × A800 80GB
# =============================================================================

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# GPU 配置：使用哪些 GPU（按需修改）
export CUDA_VISIBLE_DEVICES=0,6
NPROC_PER_NODE=2  # 与 CUDA_VISIBLE_DEVICES 中的 GPU 数量一致

# DeepSpeed 配置
# 8B 模型 + 2 卡 A800 80GB → ZeRO-3（模型参数分片到两张卡）
deepspeed=./scripts/zero3.json

# 模型路径（本地路径）
llm=/path/to/Qwen3-VL-2B-Instruct

# 训练超参数
lr=1e-5              # Qwen3 系列推荐 1e-5
batch_size=2         # video_max_frames=30 时显存占用大，per_device 设为 1
grad_accum_steps=8  # 等效 batch_size = 1 * 2卡 * 16 = 32

# 训练入口
entry_file=qwenvl/train/train_qwen.py

# 数据集名称（对应 data/__init__.py 中注册的名称）
datasets=dar_emotion_sft

# 输出路径
run_name="qwen3vl-2b-dar-emotion-sft"
output_dir=/path/to/outputs/qwen3vl-2b-dar-sft

# 训练参数
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --video_max_frames 30 \
    --video_min_frames 1 \
    --video_fps 2 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to none"

# 切到项目根目录（entry_file 是相对于此目录的）
cd "$(dirname "$0")/.."

# 启动训练
echo "=============================================="
echo "模型: ${llm}"
echo "数据集: ${datasets}"
echo "GPU数量: ${NPROC_PER_NODE}"
echo "DeepSpeed: ${deepspeed}"
echo "输出目录: ${output_dir}"
echo "=============================================="

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
