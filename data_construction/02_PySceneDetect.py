#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from dar_pipeline_common import (
    append_jsonl,
    iter_jsonl,
    load_done_video_ids,
    nearest_cut,
    normalize_segments,
    resolve_video_path,
    round_time,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align Gemini event boundaries with PySceneDetect cuts")
    parser.add_argument("--input_jsonl", default="/path/to/DAR/work/01_gemini_semantic_events.jsonl")
    parser.add_argument("--output_jsonl", default="/path/to/DAR/work/02_pyscenedetect_aligned_events.jsonl")
    parser.add_argument("--dataset_root", default="/path/to/DAR/raw_dar")
    parser.add_argument("--snap_window", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=27.0, help="PySceneDetect ContentDetector threshold")
    parser.add_argument("--min_scene_len", type=int, default=6, help="Minimum scene length in frames")
    parser.add_argument("--cv2_fallback_threshold", type=float, default=35.0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    return parser.parse_args()


def detect_cuts_pyscenedetect(video_path: str, threshold: float, min_scene_len: int) -> List[float]:
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except Exception as exc:
        raise RuntimeError(f"PySceneDetect is not importable: {exc}") from exc

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video=video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    cuts: List[float] = []
    for start_time, _end_time in scene_list[1:]:
        cuts.append(round_time(start_time.get_seconds()))
    return sorted(set(cuts))


def detect_cuts_cv2_fallback(video_path: str, threshold: float) -> List[float]:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    prev_hist = None
    cuts: List[float] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) * 100.0
            if diff >= threshold:
                cuts.append(round_time(frame_idx / fps))
        prev_hist = hist
        frame_idx += 1
    cap.release()
    return sorted(set(cuts))


def detect_visual_cuts(video_path: str, args: argparse.Namespace) -> Tuple[List[float], str, str]:
    try:
        return detect_cuts_pyscenedetect(video_path, args.threshold, args.min_scene_len), "pyscenedetect", ""
    except Exception as exc:
        try:
            cuts = detect_cuts_cv2_fallback(video_path, args.cv2_fallback_threshold)
            return cuts, "cv2_histogram_fallback", str(exc)
        except Exception as fallback_exc:
            return [], "failed", f"{exc}; fallback failed: {fallback_exc}"


def build_aligned_segments(
    semantic_segments: Sequence[Dict[str, Any]],
    visual_cuts: Sequence[float],
    duration: float,
    snap_window: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    semantic = normalize_segments(
        semantic_segments,
        duration,
        keep_keys=("event", "boundary_reason"),
    )
    if len(semantic) <= 1:
        return semantic, []

    new_boundaries = [0.0]
    snap_log: List[Dict[str, Any]] = []
    for idx, seg in enumerate(semantic[:-1]):
        original = round_time(seg["end_time"])
        snapped, delta = nearest_cut(original, visual_cuts, snap_window)
        if snapped is not None:
            boundary = snapped
            source = "visual_cut"
        else:
            boundary = original
            source = "semantic"
        new_boundaries.append(boundary)
        snap_log.append(
            {
                "boundary_index": idx,
                "original_time": original,
                "aligned_time": round_time(boundary),
                "source": source,
                "delta_to_cut": delta,
            }
        )
    new_boundaries.append(round_time(duration))

    rebuilt: List[Dict[str, Any]] = []
    for idx, seg in enumerate(semantic):
        item = dict(seg)
        item["start_time"] = new_boundaries[idx]
        item["end_time"] = new_boundaries[idx + 1]
        item["boundary_alignment"] = snap_log[idx - 1] if idx > 0 else {"source": "video_start"}
        rebuilt.append(item)

    rebuilt = normalize_segments(
        rebuilt,
        duration,
        keep_keys=("event", "boundary_reason", "boundary_alignment"),
    )
    return rebuilt, snap_log


def main() -> None:
    args = parse_args()
    items = list(iter_jsonl(args.input_jsonl))
    if args.start_index > 0 or args.end_index >= 0:
        end = args.end_index + 1 if args.end_index >= 0 else None
        items = items[args.start_index:end]

    done = load_done_video_ids(args.output_jsonl)
    print(f"PySceneDetect alignment: items={len(items)}, done={len(done)}")

    for index, item in enumerate(items):
        video_id = str(item.get("video_id", ""))
        if not video_id or video_id in done:
            continue

        record = {k: v for k, v in item.items() if k not in ("segments", "error", "attempts")}
        record["stage"] = "pyscenedetect_boundary_alignment"

        if item.get("error"):
            record["error"] = item["error"]
            append_jsonl(args.output_jsonl, record)
            continue

        video_path = resolve_video_path(str(item.get("video_path", "")), args.dataset_root, video_id)
        duration = item.get("video_duration")
        segments = item.get("segments", [])
        if not video_path or not Path(video_path).exists() or not duration or not segments:
            record["error"] = "missing_video_duration_or_segments"
            append_jsonl(args.output_jsonl, record)
            continue

        visual_cuts, detector, detector_warning = detect_visual_cuts(video_path, args)
        aligned, snap_log = build_aligned_segments(segments, visual_cuts, float(duration), args.snap_window)
        record.update(
            {
                "video_path": video_path,
                "video_duration": round_time(duration),
                "segments": aligned,
                "visual_cuts": visual_cuts,
                "snap_window": args.snap_window,
                "snap_log": snap_log,
                "scene_detector": detector,
            }
        )
        if detector_warning:
            record["scene_detector_warning"] = detector_warning
        append_jsonl(args.output_jsonl, record)

        if (index + 1) % 100 == 0:
            print(f"Aligned {index + 1}/{len(items)}")

    print(f"Done. Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
