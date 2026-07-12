# DAR-R1 GRPO Training

This folder contains the reinforcement learning stage corresponding to Sec. 4
of the camera-ready paper.

## Paper-aligned configuration

- Backbone: `Qwen2.5-VL-3B`
- Initialization: cold-start SFT checkpoint from
  `/path/to/outputs/dar_sft_qwen25vl_3b`
- Training data source:
  `/path/to/DAR/train.jsonl`
- Converted ms-swift GRPO data:
  `/path/to/DAR/train_qwen25vl_ms_grpo.jsonl`
- Trainable modules: LLM and aligner
- Frozen modules: vision encoder
- Epochs: `1`
- Learning rate: `2e-6`
- Reward functions:
  `dar_struct`, `dar_count`, `dar_seg`, `dar_emo`, `dar_reason`
- Reward weights:
  `0.10`, `0.25`, `0.25`, `0.25`, `0.15`

Run:

```bash
bash train_qwen2.5vl_grpo_dar.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_NAME=/path/to/dar_sft_checkpoint \
OUTPUT_DIR=/path/to/dar_r1_output \
bash train_qwen2.5vl_grpo_dar.sh
```

Set `PREPARE_DATA=0` only if the converted GRPO JSONL already exists.

`train_qwen3vl_grpo_dar.sh` is retained as a legacy experiment launcher and is
not the Sec. 4 paper-aligned DAR-R1 entry point.
