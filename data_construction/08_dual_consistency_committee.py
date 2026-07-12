#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

from dar_pipeline_common import iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-consistency committee aggregation")
    parser.add_argument("--annotations_jsonl", default="/path/to/DAR/work/05_qwen3vl_affect_reasoning.jsonl")
    parser.add_argument("--qwen_judge_jsonl", default="/path/to/DAR/work/06_qwen3omni_judge.jsonl")
    parser.add_argument("--internvl_judge_jsonl", default="/path/to/DAR/work/07_internvl35_judge.jsonl")
    parser.add_argument("--output_jsonl", default="/path/to/DAR/work/08_committee_verified_annotations.jsonl")
    parser.add_argument("--rewrite_manifest_jsonl", default="/path/to/DAR/work/08_rewrite_manifest.jsonl")
    parser.add_argument("--pass_threshold", type=float, default=3.5)
    return parser.parse_args()


def load_by_video(path: str) -> Dict[str, Dict[str, Any]]:
    return {str(item.get("video_id")): item for item in iter_jsonl(path) if item.get("video_id") is not None}


def segment_judgement_map(judge: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for item in judge.get("segment_judgements", []) or []:
        try:
            out[int(item.get("segment_index"))] = item
        except Exception:
            continue
    return out


def combine_segment_feedback(
    seg_idx: int,
    qwen_map: Dict[int, Dict[str, Any]],
    intern_map: Dict[int, Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    q = qwen_map.get(seg_idx, {})
    i = intern_map.get(seg_idx, {})
    q_score = float(q.get("average_score", 0.0) or 0.0)
    i_score = float(i.get("average_score", 0.0) or 0.0)
    available = [score for score, source in ((q_score, q), (i_score, i)) if source]
    avg = sum(available) / len(available) if available else 0.0
    return {
        "segment_index": seg_idx,
        "qwen3omni_score": q_score if q else None,
        "internvl35_score": i_score if i else None,
        "committee_score": avg,
        "passed": avg >= threshold and bool(available),
        "qwen3omni_feedback": q.get("feedback", ""),
        "internvl35_feedback": i.get("feedback", ""),
        "rewrite_suggestion": " ".join(
            part
            for part in (
                q.get("rewrite_suggestion", ""),
                i.get("rewrite_suggestion", ""),
            )
            if part
        ).strip(),
    }


def main() -> None:
    args = parse_args()
    annotations = load_by_video(args.annotations_jsonl)
    qwen = load_by_video(args.qwen_judge_jsonl)
    internvl = load_by_video(args.internvl_judge_jsonl)

    verified_records: List[Dict[str, Any]] = []
    rewrite_records: List[Dict[str, Any]] = []

    for video_id, annotation in annotations.items():
        qwen_map = segment_judgement_map(qwen.get(video_id, {}))
        intern_map = segment_judgement_map(internvl.get(video_id, {}))
        segment_feedback = []
        failed_indices = []
        for idx, segment in enumerate(annotation.get("segments", [])):
            feedback = combine_segment_feedback(idx, qwen_map, intern_map, args.pass_threshold)
            segment_feedback.append(feedback)
            if not feedback["passed"]:
                failed_indices.append(idx)
                rewrite_records.append(
                    {
                        "video_id": video_id,
                        "video_path": annotation.get("video_path", ""),
                        "video_duration": annotation.get("video_duration"),
                        "top3_emotions": annotation.get("top3_emotions", []),
                        "segment_index": idx,
                        "segment": segment,
                        "committee_feedback": feedback,
                    }
                )

        scores = [x["committee_score"] for x in segment_feedback if x["committee_score"] is not None]
        verified = dict(annotation)
        verified["committee_feedback"] = segment_feedback
        verified["committee_score"] = sum(scores) / max(1, len(scores))
        verified["committee_passed"] = not failed_indices
        verified["failed_segment_indices"] = failed_indices
        verified_records.append(verified)

    write_jsonl(args.output_jsonl, verified_records)
    write_jsonl(args.rewrite_manifest_jsonl, rewrite_records)
    print(f"Verified annotations: {len(verified_records)} -> {args.output_jsonl}")
    print(f"Rewrite manifest: {len(rewrite_records)} -> {args.rewrite_manifest_jsonl}")


if __name__ == "__main__":
    main()

