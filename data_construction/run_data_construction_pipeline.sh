#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work}"

DATASET_ROOT="${DATASET_ROOT:-/path/to/DAR/raw_dar}"
METADATA_JSON="${METADATA_JSON:-${DATASET_ROOT}/metadata.json}"
LABELS_JSON="${LABELS_JSON:-${DATASET_ROOT}/all_labels.json}"
SELECTED_IDS_JSON="${SELECTED_IDS_JSON:-}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"

QWEN3VL_MODEL="${QWEN3VL_MODEL:-/path/to/Qwen3-VL-32B-Instruct}"
QWEN3_OMNI_MODEL="${QWEN3_OMNI_MODEL:-${QWEN3VL_MODEL}}"
INTERNVL35_MODEL="${INTERNVL35_MODEL:-/path/to/InternVL3_5-38B-HF}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_GEMINI="${RUN_GEMINI:-1}"
RUN_SCENE_ALIGN="${RUN_SCENE_ALIGN:-1}"
RUN_INTERNVL_REFINE="${RUN_INTERNVL_REFINE:-1}"
RUN_DESCRIPTION="${RUN_DESCRIPTION:-1}"
RUN_GROUNDING="${RUN_GROUNDING:-1}"
RUN_REASONING="${RUN_REASONING:-1}"
RUN_QWEN_JUDGE="${RUN_QWEN_JUDGE:-1}"
RUN_INTERNVL_JUDGE="${RUN_INTERNVL_JUDGE:-1}"
RUN_COMMITTEE="${RUN_COMMITTEE:-1}"

mkdir -p "${WORK_DIR}"

COMMON_RANGE_ARGS=(--start_index "${START_INDEX}" --end_index "${END_INDEX}")
COMMON_DATA_ARGS=(--dataset_root "${DATASET_ROOT}" --metadata_json "${METADATA_JSON}" --labels_json "${LABELS_JSON}")

if [[ "${RUN_PREPROCESS}" == "1" ]]; then
  python "${SCRIPT_DIR}/00_dar_preprocess_filter.py" \
    "${COMMON_DATA_ARGS[@]}" \
    --selected_ids_json "${SELECTED_IDS_JSON}" \
    --output_jsonl "${WORK_DIR}/00_preprocessed_manifest.jsonl" \
    "${COMMON_RANGE_ARGS[@]}"
fi

if [[ "${RUN_GEMINI}" == "1" ]]; then
  python "${SCRIPT_DIR}/01_gemini2.5pro.py" \
    "${COMMON_DATA_ARGS[@]}" \
    --input_manifest "${WORK_DIR}/00_preprocessed_manifest.jsonl" \
    --selected_ids_json "${SELECTED_IDS_JSON}" \
    --output_jsonl "${WORK_DIR}/01_gemini_semantic_events.jsonl" \
    "${COMMON_RANGE_ARGS[@]}"
fi

if [[ "${RUN_SCENE_ALIGN}" == "1" ]]; then
  python "${SCRIPT_DIR}/02_PySceneDetect.py" \
    --dataset_root "${DATASET_ROOT}" \
    --input_jsonl "${WORK_DIR}/01_gemini_semantic_events.jsonl" \
    --output_jsonl "${WORK_DIR}/02_pyscenedetect_aligned_events.jsonl" \
    "${COMMON_RANGE_ARGS[@]}"
fi

if [[ "${RUN_INTERNVL_REFINE}" == "1" ]]; then
  python "${SCRIPT_DIR}/03_internvl35_event_integrity_refine.py" \
    --model_path "${INTERNVL35_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --input_jsonl "${WORK_DIR}/02_pyscenedetect_aligned_events.jsonl" \
    --output_jsonl "${WORK_DIR}/03_internvl_refined_events.jsonl" \
    "${COMMON_RANGE_ARGS[@]}"
fi

if [[ "${RUN_DESCRIPTION}" == "1" ]]; then
  python "${SCRIPT_DIR}/04_qwen3vl_differential_description.py" \
    --model_path "${QWEN3VL_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --input_jsonl "${WORK_DIR}/03_internvl_refined_events.jsonl" \
    --output_jsonl "${WORK_DIR}/04_qwen3vl_descriptions.jsonl" \
    "${COMMON_RANGE_ARGS[@]}"
fi

if [[ "${RUN_GROUNDING}" == "1" ]]; then
  python "${SCRIPT_DIR}/04b_grounding_dino_description_filter.py" \
    --dataset_root "${DATASET_ROOT}" \
    --input_jsonl "${WORK_DIR}/04_qwen3vl_descriptions.jsonl" \
    --output_jsonl "${WORK_DIR}/04b_grounded_descriptions.jsonl" \
    "${COMMON_RANGE_ARGS[@]}"
fi

if [[ "${RUN_REASONING}" == "1" ]]; then
  python "${SCRIPT_DIR}/05_qwen3vl_stream_affect_reason.py" \
    --model_path "${QWEN3VL_MODEL}" \
    "${COMMON_DATA_ARGS[@]}" \
    --input_jsonl "${WORK_DIR}/04b_grounded_descriptions.jsonl" \
    --output_jsonl "${WORK_DIR}/05_qwen3vl_affect_reasoning.jsonl" \
    "${COMMON_RANGE_ARGS[@]}"
fi

if [[ "${RUN_QWEN_JUDGE}" == "1" ]]; then
  python "${SCRIPT_DIR}/06_qwen3omni_quality_judge.py" \
    --model_path "${QWEN3_OMNI_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --input_jsonl "${WORK_DIR}/05_qwen3vl_affect_reasoning.jsonl" \
    --output_jsonl "${WORK_DIR}/06_qwen3omni_judge.jsonl" \
    "${COMMON_RANGE_ARGS[@]}"
fi

if [[ "${RUN_INTERNVL_JUDGE}" == "1" ]]; then
  python "${SCRIPT_DIR}/07_internvl35_quality_judge.py" \
    --model_path "${INTERNVL35_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --input_jsonl "${WORK_DIR}/05_qwen3vl_affect_reasoning.jsonl" \
    --output_jsonl "${WORK_DIR}/07_internvl35_judge.jsonl" \
    "${COMMON_RANGE_ARGS[@]}"
fi

if [[ "${RUN_COMMITTEE}" == "1" ]]; then
  python "${SCRIPT_DIR}/08_dual_consistency_committee.py" \
    --annotations_jsonl "${WORK_DIR}/05_qwen3vl_affect_reasoning.jsonl" \
    --qwen_judge_jsonl "${WORK_DIR}/06_qwen3omni_judge.jsonl" \
    --internvl_judge_jsonl "${WORK_DIR}/07_internvl35_judge.jsonl" \
    --output_jsonl "${WORK_DIR}/08_committee_verified_annotations.jsonl" \
    --rewrite_manifest_jsonl "${WORK_DIR}/08_rewrite_manifest.jsonl"
fi

echo "DAR data-construction pipeline finished. Outputs are in ${WORK_DIR}"
