#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dar_pipeline_common import (
    duration_from_metadata,
    file_from_metadata,
    load_json,
    resolve_video_path,
    top3_for_video,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAR preprocessing filter")
    parser.add_argument("--dataset_root", default="/path/to/DAR/raw_dar")
    parser.add_argument("--metadata_json", default="/path/to/DAR/raw_dar/metadata.json")
    parser.add_argument("--labels_json", default="/path/to/DAR/raw_dar/all_labels.json")
    parser.add_argument("--selected_ids_json", default="", help="Optional JSON list of video ids to process.")
    parser.add_argument("--output_jsonl", default="/path/to/DAR/work/00_preprocessed_manifest.jsonl")
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--phash_hamming_threshold", type=float, default=4.0)
    parser.add_argument("--flow_threshold", type=float, default=0.25)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--skip_static_check", action="store_true")
    return parser.parse_args()


def average_hash(gray, hash_size: int = 8) -> int:
    import cv2

    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = resized.mean()
    bits = resized > mean
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def sample_frames(video_path: str, num_frames: int) -> List[Any]:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return []

    indices = np.linspace(0, frame_count - 1, num=max(2, num_frames), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def static_motion_scores(video_path: str, num_frames: int) -> Tuple[float, float]:
    import cv2

    frames = sample_frames(video_path, num_frames)
    if len(frames) < 2:
        return 0.0, 0.0

    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    hashes = [average_hash(gray) for gray in grays]
    hash_dists = [hamming_distance(a, b) for a, b in zip(hashes, hashes[1:])]

    flow_values = []
    for prev, cur in zip(grays, grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_values.append(float(mag.mean()))

    avg_hash = sum(hash_dists) / max(1, len(hash_dists))
    avg_flow = sum(flow_values) / max(1, len(flow_values))
    return float(avg_hash), float(avg_flow)


def load_selected_ids(path: str) -> Optional[List[str]]:
    if not path:
        return None
    ids = load_json(path)
    if not isinstance(ids, list):
        raise ValueError("--selected_ids_json must point to a JSON list.")
    return [str(x) for x in ids]


def iter_candidate_ids(metadata: Dict[str, Any], selected_ids: Optional[List[str]]) -> List[str]:
    if selected_ids is not None:
        return selected_ids
    return sorted(metadata.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))


def main() -> None:
    args = parse_args()
    metadata = load_json(args.metadata_json)
    labels = load_json(args.labels_json) if args.labels_json and Path(args.labels_json).exists() else {}
    selected_ids = load_selected_ids(args.selected_ids_json)

    ids = iter_candidate_ids(metadata, selected_ids)
    if args.start_index > 0 or args.end_index >= 0:
        end = args.end_index + 1 if args.end_index >= 0 else None
        ids = ids[args.start_index:end]

    records: List[Dict[str, Any]] = []
    for video_id in ids:
        duration = duration_from_metadata(metadata, video_id)
        raw_file = file_from_metadata(metadata, video_id) or f"videos/{video_id}.mp4"
        video_path = resolve_video_path(raw_file, args.dataset_root, video_id)
        exists = bool(video_path and Path(video_path).exists())

        keep = exists and duration is not None and duration >= args.min_duration
        drop_reason = ""
        avg_hash = None
        avg_flow = None

        if not exists:
            drop_reason = "missing_video"
        elif duration is None:
            drop_reason = "missing_duration"
        elif duration < args.min_duration:
            drop_reason = "duration_lt_minimum"
        elif not args.skip_static_check:
            try:
                avg_hash, avg_flow = static_motion_scores(video_path, args.num_frames)
                is_static = (
                    avg_hash <= args.phash_hamming_threshold
                    and avg_flow <= args.flow_threshold
                )
                if is_static:
                    keep = False
                    drop_reason = "static_or_slideshow"
            except Exception as exc:
                keep = False
                drop_reason = f"static_check_failed: {exc}"

        records.append(
            {
                "video_id": str(video_id),
                "video_path": video_path,
                "video_path_raw": raw_file,
                "video_duration": duration,
                "top3_emotions": top3_for_video(video_id, metadata, labels),
                "keep": bool(keep),
                "drop_reason": drop_reason,
                "static_metrics": {
                    "avg_phash_hamming": avg_hash,
                    "avg_optical_flow": avg_flow,
                },
            }
        )

    write_jsonl(args.output_jsonl, records)
    kept = sum(1 for item in records if item["keep"])
    print(f"Saved {len(records)} records to {args.output_jsonl}; kept={kept}, dropped={len(records) - kept}")


if __name__ == "__main__":
    main()
