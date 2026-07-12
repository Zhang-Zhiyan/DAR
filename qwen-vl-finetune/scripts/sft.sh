#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${#GPU_IDS[@]}"
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-VL-3B-Instruct}"
DATASETS="${DATASETS:-dar_sft}"
export DAR_SFT_JSONL="${DAR_SFT_JSONL:-/path/to/DAR/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/outputs/dar_sft_qwen25vl_3b}"
RUN_NAME="${RUN_NAME:-dar-sft-qwen25vl-3b}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${SCRIPT_DIR}/zero3.json}"
ENTRY_FILE="${ENTRY_FILE:-qwenvl/train/train_qwen.py}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-0.5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-8192}"

MAX_PIXELS="${MAX_PIXELS:-50176}"
MIN_PIXELS="${MIN_PIXELS:-784}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-30}"
VIDEO_MIN_FRAMES="${VIDEO_MIN_FRAMES:-4}"
VIDEO_FPS="${VIDEO_FPS:-2}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-200704}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-50176}"

if [[ "${VALIDATE_DATA:-1}" == "1" ]]; then
    python tools/validate_dar_sft_data.py --dataset "${DATASETS}"
fi

args=(
    --deepspeed "${DEEPSPEED_CONFIG}"
    --model_name_or_path "${MODEL_PATH}"
    --dataset_use "${DATASETS}"
    --data_flatten True
    --tune_mm_vision False
    --tune_mm_mlp True
    --tune_mm_llm True
    --bf16
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "$((PER_DEVICE_TRAIN_BATCH_SIZE * 2))"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --max_pixels "${MAX_PIXELS}"
    --min_pixels "${MIN_PIXELS}"
    --video_max_frames "${VIDEO_MAX_FRAMES}"
    --video_min_frames "${VIDEO_MIN_FRAMES}"
    --video_fps "${VIDEO_FPS}"
    --video_max_pixels "${VIDEO_MAX_PIXELS}"
    --video_min_pixels "${VIDEO_MIN_PIXELS}"
    --eval_strategy no
    --save_strategy steps
    --save_steps 1000
    --save_total_limit 1
    --learning_rate "${LEARNING_RATE}"
    --weight_decay 0
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type cosine
    --optim adamw_torch
    --logging_steps 1
    --model_max_length "${MODEL_MAX_LENGTH}"
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --report_to none
    --run_name "${RUN_NAME}"
)

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${ENTRY_FILE}" "${args[@]}"
