#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from dar_pipeline_common import append_jsonl, extract_json_object, iter_jsonl, load_done_video_ids, resolve_video_path

warnings.filterwarnings("ignore")
logging.getLogger("lmdeploy").setLevel(logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InternVL3.5 quality judge for DAR annotations")
    parser.add_argument("--model_path", default="/path/to/InternVL3_5-38B-HF")
    parser.add_argument("--input_jsonl", default="/path/to/DAR/work/05_qwen3vl_affect_reasoning.jsonl")
    parser.add_argument("--output_jsonl", default="/path/to/DAR/work/07_internvl35_judge.jsonl")
    parser.add_argument("--dataset_root", default="/path/to/DAR/raw_dar")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--session_len", type=int, default=65536)
    parser.add_argument("--pass_threshold", type=float, default=3.5)
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


def extract_segment_frames(video_path: str, start: float, end: float, num_frames: int) -> List[Image.Image]:
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps() or 30.0)
    start_idx = max(0, int(float(start) * fps))
    end_idx = min(len(vr) - 1, max(start_idx + 1, int(float(end) * fps)))
    indices = np.linspace(start_idx, end_idx, num=max(2, num_frames), dtype=int)
    return [Image.fromarray(vr[int(idx)].asnumpy()) for idx in indices]


def build_prompt(item: Dict[str, Any], segment: Dict[str, Any], segment_index: int, num_frames: int) -> str:
    from lmdeploy.vl.constants import IMAGE_TOKEN

    frame_prefix = "".join([f"Frame{i + 1}: {IMAGE_TOKEN}\n" for i in range(num_frames)])
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

    return f"""{frame_prefix}
You are an expert evaluator for viewer-centric dynamic affective reasoning.

Target segment:
[{segment['start_time']:.1f}, {segment['end_time']:.1f}]

Question:
"What emotion would a viewer feel in this segment, and why?"

Prediction:
{prediction_json}

{prev_text}
Evaluate the predicted emotion-reason pair from 0 to 5 on:
1. visual_grounding
2. causal_logic
3. viewer_centricity
4. temporal_consistency
5. answer_consistency

Rules:
- Viewer emotion is not character emotion.
- Reason must cite visible content in the target interval.
- Consider preceding context for temporal consistency.

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


def judge_video(pipe, item: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    from lmdeploy import GenerationConfig

    judgements = []
    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0.1, top_p=0.95)
    for idx, segment in enumerate(item.get("segments", [])):
        frames = extract_segment_frames(item["video_path"], segment["start_time"], segment["end_time"], args.num_frames)
        prompt = build_prompt(item, segment, idx, len(frames))
        response = pipe((prompt, frames), gen_config=gen_config)
        raw_text = response.text if hasattr(response, "text") else str(response)
        judgement = parse_judgement(raw_text, idx)
        judgement["passed"] = judgement.get("average_score", 0.0) >= args.pass_threshold
        judgements.append(judgement)

    avg = sum(j.get("average_score", 0.0) for j in judgements) / max(1, len(judgements))
    return {
        "video_id": item.get("video_id"),
        "video_path": item.get("video_path"),
        "judge": "internvl35",
        "average_score": avg,
        "passed": all(j.get("passed", False) for j in judgements),
        "segment_judgements": judgements,
    }


def main() -> None:
    args = parse_args()
    init_ray()

    from lmdeploy import PytorchEngineConfig, pipeline

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

    print(f"InternVL3.5 judge: pending={len(pending)}, done={len(done)}")
    if not pending:
        return

    pipe = pipeline(args.model_path, backend_config=PytorchEngineConfig(session_len=args.session_len, tp=args.tp))

    for item in tqdm(pending, desc="InternVL judge"):
        try:
            record = judge_video(pipe, item, args)
        except Exception as exc:
            record = {"video_id": item.get("video_id"), "error": f"judge_failed: {exc}"}
        append_jsonl(args.output_jsonl, record)

    print(f"Done. Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
