#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from dar_pipeline_common import append_jsonl, extract_json_object, iter_jsonl, load_done_video_ids, resolve_video_path


DEFAULT_QWEN_UTILS = "/path/to/qwen-vl-utils/src"
if DEFAULT_QWEN_UTILS not in sys.path:
    sys.path.append(DEFAULT_QWEN_UTILS)
from qwen_vl_utils import process_vision_info  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-Omni quality judge for DAR annotations")
    parser.add_argument("--model_path", default=os.environ.get("QWEN3_OMNI_MODEL", "/path/to/Qwen3-VL-32B-Instruct"))
    parser.add_argument("--input_jsonl", default="/path/to/DAR/work/05_qwen3vl_affect_reasoning.jsonl")
    parser.add_argument("--output_jsonl", default="/path/to/DAR/work/06_qwen3omni_judge.jsonl")
    parser.add_argument("--dataset_root", default="/path/to/DAR/raw_dar")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--pass_threshold", type=float, default=3.5)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    return parser.parse_args()


def build_judge_prompt(item: Dict[str, Any], segment: Dict[str, Any], segment_index: int) -> str:
    previous = item.get("segments", [])[segment_index - 1] if segment_index > 0 else None
    prev_text = ""
    if previous:
        prev_text = f"""Previous segment context:
- Time: [{previous['start_time']:.1f}s, {previous['end_time']:.1f}s]
- Emotion: {previous.get('emotion', '')}
- Reason: {previous.get('reason', '')}
"""
    prediction_json = json.dumps(
        {"emotion": segment.get("emotion", ""), "reason": segment.get("reason", "")},
        ensure_ascii=False,
        indent=2,
    )

    return f"""You are an expert evaluator for viewer-centric dynamic affective reasoning.

Inputs:
1. A silent video segment.
2. A target temporal segment: [{segment['start_time']:.1f}, {segment['end_time']:.1f}].
3. A question: "What emotion would a viewer feel in this segment, and why?"
4. A model prediction containing a predicted emotion and reason.

Rules:
- The emotion refers to the VIEWER's induced emotion, not the emotion of characters.
- The reason should be grounded in visible evidence.
- The answer should focus on the target temporal segment.
- If this is not the first segment, consider temporal consistency with preceding context.

{prev_text}
Current prediction:
{prediction_json}

Evaluate the predicted emotion-reason pair from 0 to 5 on each dimension:
- visual_grounding: Is the reason supported by visible content in the segment?
- causal_logic: Does the reason explain why the visual event induces the viewer emotion?
- viewer_centricity: Does it focus on viewer emotion rather than character emotion?
- temporal_consistency: Is it consistent with the specified time interval and preceding context?
- answer_consistency: Are the emotion label and reason mutually consistent?

Return ONLY JSON:
{{
  "segment_index": {segment_index},
  "visual_grounding": <integer 0-5>,
  "causal_logic": <integer 0-5>,
  "viewer_centricity": <integer 0-5>,
  "temporal_consistency": <integer 0-5>,
  "answer_consistency": <integer 0-5>,
  "average_score": <float>,
  "feedback": "<specific actionable feedback>",
  "rewrite_suggestion": "<specific changes needed if score is low>"
}}"""


def make_messages(video_path: str, segment: Dict[str, Any], prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": os.path.abspath(video_path),
                    "video_start": float(segment["start_time"]),
                    "video_end": float(segment["end_time"]),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_input(processor: AutoProcessor, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data: Dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def parse_judgement(raw_text: str, segment_index: int) -> Dict[str, Any]:
    parsed = extract_json_object(raw_text) or {}
    dims = ["visual_grounding", "causal_logic", "viewer_centricity", "temporal_consistency", "answer_consistency"]
    scores = []
    for dim in dims:
        try:
            scores.append(float(parsed.get(dim, 0)))
        except Exception:
            scores.append(0.0)
    avg = sum(scores) / len(scores) if scores else 0.0
    parsed["segment_index"] = int(parsed.get("segment_index", segment_index))
    parsed["average_score"] = float(parsed.get("average_score", avg) or avg)
    parsed.setdefault("feedback", "")
    parsed.setdefault("rewrite_suggestion", "")
    return parsed


def judge_video(
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    item: Dict[str, Any],
    pass_threshold: float,
) -> Dict[str, Any]:
    video_path = item["video_path"]
    judgements = []
    for idx, segment in enumerate(item.get("segments", [])):
        prompt = build_judge_prompt(item, segment, idx)
        messages = make_messages(video_path, segment, prompt)
        model_input = build_input(processor, messages)
        output = llm.generate([model_input], sampling_params=sampling_params)[0]
        raw_text = output.outputs[0].text
        judgement = parse_judgement(raw_text, idx)
        judgement["passed"] = judgement.get("average_score", 0.0) >= pass_threshold
        judgements.append(judgement)
        gc.collect()

    avg = sum(j.get("average_score", 0.0) for j in judgements) / max(1, len(judgements))
    return {
        "video_id": item.get("video_id"),
        "video_path": item.get("video_path"),
        "judge": "qwen3_omni",
        "average_score": avg,
        "passed": all(j.get("passed", False) for j in judgements),
        "segment_judgements": judgements,
    }


def main() -> None:
    args = parse_args()
    items = list(iter_jsonl(args.input_jsonl))
    if args.start_index > 0 or args.end_index >= 0:
        end = args.end_index + 1 if args.end_index >= 0 else None
        items = items[args.start_index:end]

    done = load_done_video_ids(args.output_jsonl)
    pending = []
    for item in items:
        video_id = str(item.get("video_id", ""))
        if not video_id or video_id in done:
            continue
        if item.get("error"):
            append_jsonl(args.output_jsonl, item)
            continue
        video_path = resolve_video_path(str(item.get("video_path", "")), args.dataset_root, video_id)
        if not video_path or not Path(video_path).exists():
            append_jsonl(args.output_jsonl, {"video_id": video_id, "error": "missing_video"})
            continue
        prepared = dict(item)
        prepared["video_path"] = video_path
        pending.append(prepared)

    print(f"Qwen3-Omni judge: pending={len(pending)}, done={len(done)}")
    if not pending:
        return

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        max_model_len=args.max_model_len,
        seed=1234,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_tokens,
        stop_token_ids=[],
    )

    for item in tqdm(pending, desc="Qwen judge"):
        try:
            record = judge_video(llm, processor, sampling_params, item, args.pass_threshold)
        except Exception as exc:
            record = {"video_id": item.get("video_id"), "error": f"judge_failed: {exc}"}
        append_jsonl(args.output_jsonl, record)

    print(f"Done. Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
