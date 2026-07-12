#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from dar_pipeline_common import (
    append_jsonl,
    extract_json_object,
    iter_jsonl,
    load_done_video_ids,
    normalize_segments,
    resolve_video_path,
    round_time,
)

warnings.filterwarnings("ignore")
logging.getLogger("lmdeploy").setLevel(logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InternVL3.5 event-integrity refinement for DAR")
    parser.add_argument("--model_path", default="/path/to/InternVL3_5-38B-HF")
    parser.add_argument("--input_jsonl", default="/path/to/DAR/work/02_pyscenedetect_aligned_events.jsonl")
    parser.add_argument("--output_jsonl", default="/path/to/DAR/work/03_internvl_refined_events.jsonl")
    parser.add_argument("--dataset_root", default="/path/to/DAR/raw_dar")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--session_len", type=int, default=65536)
    parser.add_argument("--max_refine_rounds", type=int, default=2)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    return parser.parse_args()


def init_ray() -> None:
    try:
        import ray
    except Exception:
        return
    os.environ.pop("RAY_ADDRESS", None)
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["RAY_LOG_TO_STDERR"] = "0"
    if not ray.is_initialized():
        try:
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level="error",
                log_to_driver=False,
                _system_config={
                    "local_fs_capacity_threshold": 0.99,
                    "object_spilling_config": None,
                },
            )
        except Exception:
            ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="error", log_to_driver=False)


def extract_video_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    if total_frames <= 0:
        return []
    indices = np.linspace(0, total_frames - 1, num=max(2, num_frames), dtype=int)
    return [Image.fromarray(vr[int(idx)].asnumpy()) for idx in indices]


def format_segments(segments: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for i, seg in enumerate(segments):
        lines.append(
            f"Segment {i}: [{float(seg['start_time']):.1f}s, {float(seg['end_time']):.1f}s] "
            f"event=\"{seg.get('event', '')}\""
        )
    return "\n".join(lines)


def build_prompt(video_duration: float, segments: Sequence[Dict[str, Any]], num_frames: int) -> str:
    from lmdeploy.vl.constants import IMAGE_TOKEN

    frame_prefix = "".join([f"Frame{i + 1}: {IMAGE_TOKEN}\n" for i in range(num_frames)])
    segment_text = format_segments(segments)
    return f"""{frame_prefix}
You are verifying event-aligned segmentation for a silent short video.

Goal:
Each segment should be one complete visual/narrative event stimulus. The event
should not start in the middle of an action, end before the action finishes, or
contain two clearly separable events.

Current candidate segments:
{segment_text}

Video duration: {video_duration:.1f}s.

Decide whether the segmentation is already event-complete. If not, refine it by:
- merging adjacent fragments that only form a complete event together;
- splitting a segment that contains multiple complete events;
- preserving boundaries that already align with coherent event changes.

Rules:
1. Output must cover [0.0, {video_duration:.1f}] with continuous timestamps.
2. Do not output emotions or reasons.
3. Keep the segment count compact; avoid tiny fragments unless a clear event starts.
4. Use one decimal place for timestamps.

Return ONLY JSON:
{{
  "all_segments_complete": true or false,
  "refined_segments": [
    {{"start_time": 0.0, "end_time": <float>, "event": "<visual event description>"}}
  ],
  "refinement_notes": "<brief explanation of merge/split decisions>"
}}"""


def parse_response(text: str) -> Optional[Dict[str, Any]]:
    return extract_json_object(text)


def refine_batch(pipe, batch: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    from lmdeploy import GenerationConfig

    valid_inputs: List[Tuple[str, List[Image.Image]]] = []
    valid_indices: List[int] = []
    results: List[Optional[Dict[str, Any]]] = [None] * len(batch)

    for idx, item in enumerate(batch):
        try:
            frames = extract_video_frames(item["video_path"], args.num_frames)
            prompt = build_prompt(item["video_duration"], item["segments"], len(frames))
            valid_inputs.append((prompt, frames))
            valid_indices.append(idx)
        except Exception as exc:
            results[idx] = {"error": f"input_prepare_failed: {exc}"}

    if valid_inputs:
        gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0.1, top_p=0.95)
        try:
            responses = pipe(valid_inputs, gen_config=gen_config)
        except Exception as exc:
            for idx in valid_indices:
                results[idx] = {"error": f"inference_failed: {exc}"}
        else:
            for response, original_idx in zip(responses, valid_indices):
                raw_text = response.text if hasattr(response, "text") else str(response)
                parsed = parse_response(raw_text)
                if not parsed:
                    results[original_idx] = {"error": "json_parse_failed", "raw_output": raw_text[:1000]}
                else:
                    results[original_idx] = parsed

    return [r or {"error": "unknown_error"} for r in results]


def apply_refinement(item: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    duration = float(item["video_duration"])
    refined = parsed.get("refined_segments")
    if not isinstance(refined, list) or not refined:
        refined = item["segments"]
    refined_segments = normalize_segments(refined, duration, keep_keys=("event",))
    out = dict(item)
    out["segments"] = refined_segments
    out.setdefault("integrity_rounds", []).append(parsed)
    return out


def main() -> None:
    args = parse_args()
    init_ray()

    from lmdeploy import PytorchEngineConfig, pipeline

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
        prepared["video_duration"] = round_time(prepared.get("video_duration", 0))
        prepared["segments"] = normalize_segments(prepared["segments"], prepared["video_duration"], keep_keys=("event", "boundary_reason", "boundary_alignment"))
        prepared["integrity_rounds"] = []
        pending.append(prepared)

    print(f"InternVL3.5 event refinement: pending={len(pending)}, done={len(done)}")
    if not pending:
        return

    pipe = pipeline(args.model_path, backend_config=PytorchEngineConfig(session_len=args.session_len, tp=args.tp))

    with tqdm(total=len(pending), desc="Refining") as pbar:
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            current_batch = batch
            for _round_idx in range(args.max_refine_rounds):
                parsed_results = refine_batch(pipe, current_batch, args)
                next_batch = []
                for item, parsed in zip(current_batch, parsed_results):
                    if "error" in parsed:
                        item.setdefault("integrity_rounds", []).append(parsed)
                        next_batch.append(item)
                        continue
                    updated = apply_refinement(item, parsed)
                    if parsed.get("all_segments_complete") is True:
                        next_batch.append(updated)
                    else:
                        next_batch.append(updated)
                current_batch = next_batch
                if all(
                    item.get("integrity_rounds")
                    and item["integrity_rounds"][-1].get("all_segments_complete") is True
                    for item in current_batch
                ):
                    break

            for item in current_batch:
                record = dict(item)
                record["stage"] = "internvl35_event_integrity_refinement"
                append_jsonl(args.output_jsonl, record)
                pbar.update(1)

    print(f"Done. Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()

