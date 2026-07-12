#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from dar_pipeline_common import (
    append_jsonl,
    extract_json_object,
    iter_jsonl,
    load_done_video_ids,
    normalize_segments,
    resolve_video_path,
    round_time,
)


DEFAULT_QWEN_UTILS = "/path/to/qwen-vl-utils/src"
if DEFAULT_QWEN_UTILS not in sys.path:
    sys.path.append(DEFAULT_QWEN_UTILS)
from qwen_vl_utils import process_vision_info  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL differential descriptions for DAR")
    parser.add_argument("--model_path", default="/path/to/Qwen3-VL-32B-Instruct")
    parser.add_argument("--input_jsonl", default="/path/to/DAR/work/03_internvl_refined_events.jsonl")
    parser.add_argument("--output_jsonl", default="/path/to/DAR/work/04_qwen3vl_descriptions.jsonl")
    parser.add_argument("--dataset_root", default="/path/to/DAR/raw_dar")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    return parser.parse_args()


def build_prompt(
    video_id: str,
    segment: Dict[str, Any],
    segment_index: int,
    total_segments: int,
    previous_description: str,
) -> str:
    st = float(segment["start_time"])
    et = float(segment["end_time"])
    event = str(segment.get("event", "")).strip()

    if segment_index == 0:
        task = """This is the first segment. Provide a complete visual description of the segment:
- setting and background;
- main subjects and objects;
- visible actions and motion;
- visual atmosphere such as lighting, color, composition, and camera movement."""
        context = "Previous description D_{i-1}: <empty>"
    else:
        task = """This is not the first segment. Generate an incremental differential description:
- focus on what changes compared with the previous segment;
- mention new actions, changed object/subject states, scene changes, camera movement, and atmosphere shifts;
- avoid repeating unchanged background details unless they are needed to understand the change."""
        context = f"Previous description D_{{i-1}}:\n{previous_description}"

    return f"""You are generating visually grounded captions for a silent short video.

Important:
- This stage outputs visual descriptions only.
- Do not mention viewer emotions, emotion labels, affective reasons, or candidate emotions.
- Do not mention audio, music, speech, or dialogue.
- Use only visually observable evidence.

Video id: {video_id}
Current segment S_i: {segment_index + 1}/{total_segments}
Time interval: [{st:.1f}s, {et:.1f}s]
Coarse event cue: {event}

{context}

Task:
{task}

Return ONLY JSON:
{{
  "segment_index": {segment_index},
  "description": "<one detailed paragraph, no emotion labels>"
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


def parse_description(text: str) -> str:
    parsed = extract_json_object(text)
    if parsed and isinstance(parsed.get("description"), str):
        return parsed["description"].strip()
    return text.strip()


def describe_video(
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    video_id = str(item["video_id"])
    video_path = item["video_path"]
    duration = float(item["video_duration"])
    segments = normalize_segments(item["segments"], duration, keep_keys=("event", "boundary_reason", "boundary_alignment"))

    described_segments: List[Dict[str, Any]] = []
    previous_description = ""
    for seg_idx, segment in enumerate(segments):
        prompt = build_prompt(video_id, segment, seg_idx, len(segments), previous_description)
        messages = make_messages(video_path, segment, prompt)
        model_input = build_input(processor, messages)
        output = llm.generate([model_input], sampling_params=sampling_params)[0]
        raw_text = output.outputs[0].text
        description = parse_description(raw_text)

        out_seg = {
            "start_time": round_time(segment["start_time"]),
            "end_time": round_time(segment["end_time"]),
            "event": segment.get("event", ""),
            "description": description,
        }
        if "boundary_alignment" in segment:
            out_seg["boundary_alignment"] = segment["boundary_alignment"]
        described_segments.append(out_seg)
        previous_description = description
        gc.collect()

    out = {k: v for k, v in item.items() if k not in ("segments", "integrity_rounds")}
    out["stage"] = "qwen3vl_incremental_differential_captioning"
    out["segments"] = described_segments
    return out


def main() -> None:
    args = parse_args()
    items = list(iter_jsonl(args.input_jsonl))
    if args.start_index > 0 or args.end_index >= 0:
        end = args.end_index + 1 if args.end_index >= 0 else None
        items = items[args.start_index:end]

    done = load_done_video_ids(args.output_jsonl)
    pending: List[Dict[str, Any]] = []
    for item in items:
        video_id = str(item.get("video_id", ""))
        if not video_id or video_id in done:
            continue
        if item.get("error"):
            append_jsonl(args.output_jsonl, item)
            continue
        video_path = resolve_video_path(str(item.get("video_path", "")), args.dataset_root, video_id)
        if not video_path or not Path(video_path).exists() or not item.get("segments"):
            failed = dict(item)
            failed["error"] = "missing_video_or_segments"
            append_jsonl(args.output_jsonl, failed)
            continue
        prepared = dict(item)
        prepared["video_path"] = video_path
        pending.append(prepared)

    print(f"Qwen3-VL descriptions: pending={len(pending)}, done={len(done)}")
    if not pending:
        return

    tensor_parallel_size = max(1, torch.cuda.device_count())
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
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

    for item in tqdm(pending, desc="Describing"):
        try:
            record = describe_video(llm, processor, sampling_params, item)
        except Exception as exc:
            record = {
                "video_id": item.get("video_id"),
                "video_path": item.get("video_path"),
                "video_duration": item.get("video_duration"),
                "top3_emotions": item.get("top3_emotions", []),
                "error": f"description_failed: {exc}",
            }
        append_jsonl(args.output_jsonl, record)

    print(f"Done. Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()

