#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SWIFT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${SWIFT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    export NPROC_PER_NODE="${#GPU_IDS[@]}"
fi
export MASTER_PORT="${MASTER_PORT:-29501}"

MODEL_NAME="${MODEL_NAME:-/path/to/Qwen3-VL-2B-Instruct}"
SOURCE_SFT_JSONL="${SOURCE_SFT_JSONL:-/path/to/DAR/train.jsonl}"
DATA_JSONL="${DATA_JSONL:-/path/to/DAR/train_qwen25vl_ms_grpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/outputs/legacy_qwen3vl_grpo}"
PROMPT_FILE="${PROMPT_FILE:-${GRPO_DIR}/prompt.txt}"
PLUGIN_FILE="${PLUGIN_FILE:-${GRPO_DIR}/plugin/dar_plugin.py}"

if [[ "${PREPARE_DATA:-1}" == "1" ]]; then
    python "${SCRIPT_DIR}/prepare_dar_grpo_data.py" \
        --input "${SOURCE_SFT_JSONL}" \
        --output "${DATA_JSONL}" \
        --prompt "${PROMPT_FILE}"
fi

swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_NAME}" \
  --template qwen3_vl \
  --dataset "${DATA_JSONL}" \
  --train_type full \
  --freeze_vit true \
  --freeze_llm false \
  --freeze_aligner false \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-6 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0 \
  --bf16 true \
  --tf32 true \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 3 \
  --report_to none \
  --dataloader_num_workers 4 \
  --max_length 4096 \
  --max_completion_length 2400 \
  --num_generations 4 \
  --temperature 0.8 \
  --top_k 50 \
  --beta 0.04 \
  --external_plugins "${PLUGIN_FILE}" \
  --reward_funcs dar_struct dar_count dar_seg dar_emo dar_reason \
  --reward_weights 0.10 0.25 0.25 0.25 0.15 \
  --output_dir "${OUTPUT_DIR}" \
  --deepspeed zero3
