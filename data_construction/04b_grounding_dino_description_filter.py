#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image
from tqdm import tqdm

from dar_pipeline_common import append_jsonl, iter_jsonl, load_done_video_ids, resolve_video_path


STOPWORDS = {
    "the", "a", "an", "and", "or", "with", "without", "from", "into", "onto", "under",
    "over", "scene", "segment", "video", "camera", "frame", "lighting", "background",
    "foreground", "movement", "motion", "view", "shot", "color", "colors", "area",
    "side", "part", "moment", "time", "transition", "atmosphere",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounding DINO description faithfulness filter")
    parser.add_argument("--input_jsonl", default="/path/to/DAR/work/04_qwen3vl_descriptions.jsonl")
    parser.add_argument("--output_jsonl", default="/path/to/DAR/work/04b_grounded_descriptions.jsonl")
    parser.add_argument("--dataset_root", default="/path/to/DAR/raw_dar")
    parser.add_argument("--model_path", default=os.environ.get("GROUNDING_DINO_MODEL", ""))
    parser.add_argument("--score_threshold", type=float, default=0.25)
    parser.add_argument("--max_terms", type=int, default=8)
    parser.add_argument("--min_grounded_terms", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    return parser.parse_args()


def get_segment_frame(video_path: str, start: float, end: float) -> Image.Image:
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps() or 30.0)
    midpoint = max(0.0, (float(start) + float(end)) / 2.0)
    frame_idx = min(len(vr) - 1, max(0, int(midpoint * fps)))
    return Image.fromarray(vr[frame_idx].asnumpy())


def extract_candidate_terms(description: str, max_terms: int) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", description.lower())
    counts: Dict[str, int] = {}
    for word in words:
        word = word.strip("-")
        if word in STOPWORDS or len(word) < 3:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:max_terms]]


def load_detector(model_path: str):
    if not model_path:
        return None, "GROUNDING_DINO_MODEL or --model_path is not set."
    try:
        from transformers import pipeline

        return pipeline(model=model_path, task="zero-shot-object-detection"), ""
    except Exception as exc:
        return None, f"Could not load Grounding DINO detector: {exc}"


def run_grounding(detector, image: Image.Image, labels: Sequence[str], threshold: float) -> List[Dict[str, Any]]:
    if not labels:
        return []
    try:
        outputs = detector(image, candidate_labels=list(labels), threshold=threshold)
    except TypeError:
        outputs = detector(image, candidate_labels=list(labels))
    grounded = []
    for item in outputs or []:
        score = float(item.get("score", 0.0))
        if score < threshold:
            continue
        grounded.append(
            {
                "label": item.get("label", ""),
                "score": score,
                "box": item.get("box", {}),
            }
        )
    return grounded


def main() -> None:
    args = parse_args()
    items = list(iter_jsonl(args.input_jsonl))
    if args.start_index > 0 or args.end_index >= 0:
        end = args.end_index + 1 if args.end_index >= 0 else None
        items = items[args.start_index:end]

    done = load_done_video_ids(args.output_jsonl)
    detector, detector_warning = load_detector(args.model_path)

    print(f"Grounding filter: items={len(items)}, done={len(done)}, detector={'on' if detector else 'skipped'}")
    for item in tqdm(items, desc="Grounding"):
        video_id = str(item.get("video_id", ""))
        if not video_id or video_id in done:
            continue
        record = dict(item)
        if item.get("error"):
            append_jsonl(args.output_jsonl, record)
            continue

        video_path = resolve_video_path(str(item.get("video_path", "")), args.dataset_root, video_id)
        if not detector:
            record["grounding_status"] = "skipped"
            record["grounding_warning"] = detector_warning
            append_jsonl(args.output_jsonl, record)
            continue
        if not video_path or not Path(video_path).exists():
            record["grounding_status"] = "failed"
            record["grounding_warning"] = "missing_video"
            append_jsonl(args.output_jsonl, record)
            continue

        filtered_segments = []
        for seg in record.get("segments", []):
            seg = dict(seg)
            terms = extract_candidate_terms(str(seg.get("description", "")), args.max_terms)
            try:
                frame = get_segment_frame(video_path, seg["start_time"], seg["end_time"])
                grounded = run_grounding(detector, frame, terms, args.score_threshold)
                grounded_labels = {str(x.get("label", "")).lower() for x in grounded}
                seg["grounding"] = {
                    "candidate_terms": terms,
                    "grounded_terms": grounded,
                    "passed": len(grounded_labels) >= args.min_grounded_terms or not terms,
                }
            except Exception as exc:
                seg["grounding"] = {
                    "candidate_terms": terms,
                    "grounded_terms": [],
                    "passed": False,
                    "error": str(exc),
                }
            filtered_segments.append(seg)

        record["segments"] = filtered_segments
        record["grounding_status"] = "checked"
        record["stage"] = "grounding_dino_description_filter"
        append_jsonl(args.output_jsonl, record)

    print(f"Done. Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()

