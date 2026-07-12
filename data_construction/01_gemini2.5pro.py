#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from dar_pipeline_common import (
    append_jsonl,
    duration_from_metadata,
    extract_json_object,
    file_from_metadata,
    iter_jsonl,
    load_done_video_ids,
    load_json,
    normalize_segments,
    resolve_video_path,
    round_time,
    top3_for_video,
)


MODEL_NAME = "models/gemini-2.5-pro"


SEGMENTATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_duration": {"type": "number"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "event": {"type": "string"},
                    "boundary_reason": {"type": "string"},
                },
                "required": ["start_time", "end_time", "event"],
            },
        },
    },
    "required": ["video_duration", "segments"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini semantic event segmentation for DAR")
    parser.add_argument("--dataset_root", default="/path/to/DAR/raw_dar")
    parser.add_argument("--metadata_json", default="/path/to/DAR/raw_dar/metadata.json")
    parser.add_argument("--labels_json", default="/path/to/DAR/raw_dar/all_labels.json")
    parser.add_argument("--input_manifest", default="/path/to/DAR/work/00_preprocessed_manifest.jsonl")
    parser.add_argument("--selected_ids_json", default="", help="Optional JSON list used when input_manifest is absent.")
    parser.add_argument("--output_jsonl", default="/path/to/DAR/work/01_gemini_semantic_events.jsonl")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_output_tokens", type=int, default=1536)
    parser.add_argument("--min_segments", type=int, default=1)
    parser.add_argument("--max_segments", type=int, default=6)
    parser.add_argument("--max_retry", type=int, default=2)
    parser.add_argument("--sleep_sec", type=float, default=1.0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    return parser.parse_args()


def load_selected_ids(path: str) -> Optional[List[str]]:
    if not path:
        return None
    ids = load_json(path)
    if not isinstance(ids, list):
        raise ValueError("--selected_ids_json must point to a JSON list.")
    return [str(x) for x in ids]


def build_items(args: argparse.Namespace, metadata: Dict[str, Any], labels: Dict[str, Any]) -> List[Dict[str, Any]]:
    input_manifest = Path(args.input_manifest)
    if input_manifest.exists():
        items = [item for item in iter_jsonl(input_manifest) if item.get("keep", True)]
    else:
        selected = load_selected_ids(args.selected_ids_json)
        ids = selected or sorted(metadata.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        items = []
        for video_id in ids:
            raw_file = file_from_metadata(metadata, video_id) or f"videos/{video_id}.mp4"
            items.append(
                {
                    "video_id": str(video_id),
                    "video_path": resolve_video_path(raw_file, args.dataset_root, str(video_id)),
                    "video_path_raw": raw_file,
                    "video_duration": duration_from_metadata(metadata, str(video_id)),
                    "top3_emotions": top3_for_video(str(video_id), metadata, labels),
                }
            )

    if args.start_index > 0 or args.end_index >= 0:
        end = args.end_index + 1 if args.end_index >= 0 else None
        items = items[args.start_index:end]
    return items


def build_prompts(video_duration: float, min_segments: int, max_segments: int) -> Tuple[str, str]:
    dur = round_time(video_duration)
    system_prompt = f"""You are a video event segmentation tool for silent short videos.

Your ONLY task is to propose high-level semantic event boundaries.
Do NOT classify emotions.
Do NOT explain viewer feelings.
Do NOT use or invent emotion labels.

Definition:
An event segment is a temporally continuous stimulus unit with coherent visual action,
narrative state, or affective turning point. Boundaries should capture semantic shifts,
action changes, reveals, impacts, payoffs, or meaningful scene/narrative transitions.

Hard constraints:
1. Cover the complete video from 0.0 to {dur:.1f} seconds.
2. Segments must be continuous, non-overlapping, and gap-free.
3. Return between {min_segments} and {max_segments} segments.
4. Timestamps are seconds with one decimal place.
5. Each segment has start_time, end_time, event, and optional boundary_reason.

Output only a valid JSON object."""

    user_prompt = f"""VIDEO_DURATION_SECONDS = {dur:.1f}

Return JSON in this exact shape:
{{
  "video_duration": {dur:.1f},
  "segments": [
    {{
      "start_time": 0.0,
      "end_time": <float>,
      "event": "<short visual event description without emotion labels>",
      "boundary_reason": "<why this event boundary is useful, if applicable>"
    }}
  ]
}}

Remember: output event segmentation only. No emotion field, no reason field."""
    return system_prompt, user_prompt


def extract_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        parts = getattr(getattr(cand, "content", None), "parts", None) or []
        for part in parts:
            value = getattr(part, "text", None)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def parsed_response(resp: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, dict):
        return parsed, ""
    raw = extract_text(resp)
    return extract_json_object(raw), raw


def gemini_segment_one(
    client: genai.Client,
    args: argparse.Namespace,
    video_path: str,
    video_duration: float,
) -> Dict[str, Any]:
    video_bytes = Path(video_path).read_bytes()
    system_prompt, user_prompt = build_prompts(video_duration, args.min_segments, args.max_segments)

    last_error = ""
    attempts: List[Dict[str, Any]] = []
    for attempt in range(args.max_retry + 1):
        retry_hint = ""
        if attempt and last_error:
            retry_hint = f"\n\nPrevious output error: {last_error}\nReturn a corrected JSON object only."

        content = types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
                    video_metadata=types.VideoMetadata(fps=int(args.fps)),
                ),
                types.Part(text=user_prompt + retry_hint),
            ]
        )

        try:
            resp = client.models.generate_content(
                model=args.model,
                contents=content,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=512),
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_json_schema=SEGMENTATION_SCHEMA,
                ),
            )
        except Exception as exc:
            last_error = f"API call failed: {exc}"
            attempts.append({"attempt": attempt, "error": last_error})
            continue

        raw_obj, raw_text = parsed_response(resp)
        attempts.append({"attempt": attempt, "raw_text_len": len(raw_text or "")})
        try:
            if not isinstance(raw_obj, dict):
                raise ValueError("Gemini did not return a JSON object.")
            segments = raw_obj.get("segments")
            if not isinstance(segments, list) or not segments:
                raise ValueError("Missing non-empty segments list.")
            if len(segments) < args.min_segments or len(segments) > args.max_segments:
                raise ValueError(f"Segment count {len(segments)} outside [{args.min_segments}, {args.max_segments}].")
            fixed = normalize_segments(
                segments,
                video_duration,
                keep_keys=("event", "boundary_reason"),
            )
            return {
                "ok": True,
                "video_duration": round_time(video_duration),
                "segments": fixed,
                "attempts": attempts,
            }
        except Exception as exc:
            last_error = str(exc)

    return {"ok": False, "error": last_error, "attempts": attempts}


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    metadata = load_json(args.metadata_json)
    labels = load_json(args.labels_json) if args.labels_json and Path(args.labels_json).exists() else {}
    items = build_items(args, metadata, labels)
    done = load_done_video_ids(args.output_jsonl)
    client = genai.Client(api_key=api_key)

    print(f"Gemini semantic event segmentation: items={len(items)}, done={len(done)}")
    for index, item in enumerate(items):
        video_id = str(item.get("video_id") or Path(str(item.get("video_path", ""))).stem)
        if video_id in done:
            continue

        video_path = resolve_video_path(str(item.get("video_path", "")), args.dataset_root, video_id)
        duration = item.get("video_duration")
        if duration is None:
            duration = duration_from_metadata(metadata, video_id)

        if not video_path or not Path(video_path).exists() or duration is None:
            append_jsonl(
                args.output_jsonl,
                {
                    "video_id": video_id,
                    "video_path": video_path,
                    "video_duration": duration,
                    "top3_emotions": top3_for_video(video_id, metadata, labels),
                    "error": "missing_video_or_duration",
                },
            )
            continue

        result = gemini_segment_one(client, args, video_path, float(duration))
        record = {
            "video_id": video_id,
            "video_path": video_path,
            "video_path_raw": item.get("video_path_raw", ""),
            "video_duration": round_time(duration),
            "top3_emotions": item.get("top3_emotions") or top3_for_video(video_id, metadata, labels),
            "stage": "gemini_semantic_event_segmentation",
        }
        if result.get("ok"):
            record["segments"] = result["segments"]
        else:
            record["error"] = result.get("error", "unknown_error")
        record["attempts"] = result.get("attempts", [])
        append_jsonl(args.output_jsonl, record)

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)
        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{len(items)}")

    print(f"Done. Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()

