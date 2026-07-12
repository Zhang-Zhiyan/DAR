# DAR-R1 Cold-Start SFT

This folder contains the supervised fine-tuning code corresponding to Sec. 4
of the camera-ready paper.

## Paper-aligned configuration

- Backbone: `Qwen2.5-VL-3B-Instruct`
- Training data: `dar_sft`, registered in `qwenvl/data/__init__.py`
- Annotation file: `/path/to/DAR/train.jsonl`
- Split size: 13,646 training videos
- Objective: generate `segments`, where each segment contains `start_time`,
  `end_time`, `emotion`, and `reason`
- Trainable modules: LLM and aligner (`visual.merger`)
- Frozen modules: vision encoder
- Optimizer: AdamW
- Learning rate: `1e-5`
- Epochs: `0.5`
- Default launch: 4 GPUs

The canonical launch script is:

```bash
bash scripts/sft.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1 MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
OUTPUT_DIR=/path/to/output bash scripts/sft.sh
```

The script runs `tools/validate_dar_sft_data.py` before training. Set
`VALIDATE_DATA=0` only when the dataset has already been checked.

Legacy scripts in `scripts/` are retained only as local experiment references;
they are not the paper-aligned DAR-SFT entry point.
