#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.data import data_list
from qwenvl.data.data_processor import EMOTION_CANDIDATES


ALLOWED_EMOTIONS = set(EMOTION_CANDIDATES)


def load_annotations(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL records")
    return data


def parse_answer(item, index):
    conversations = item.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError(f"line {index}: missing two-turn conversations")
    if conversations[0].get("from") != "human" or conversations[1].get("from") != "gpt":
        raise ValueError(f"line {index}: expected human/gpt conversation roles")

    answer = conversations[1].get("value", "")
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {index}: assistant answer is not valid JSON: {exc}") from exc
    return parsed


def close(a, b, tol=0.15):
    return abs(float(a) - float(b)) <= tol


def validate_item(item, index):
    if not item.get("video"):
        raise ValueError(f"line {index}: missing video path")

    duration = item.get("video_duration")
    if duration is None or float(duration) <= 0:
        raise ValueError(f"line {index}: invalid video_duration={duration}")
    duration = float(duration)

    parsed = parse_answer(item, index)
    segments = parsed.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"line {index}: segments must be a non-empty list")

    previous_end = None
    previous_emotion = None
    for seg_idx, segment in enumerate(segments):
        for key in ("start_time", "end_time", "emotion", "reason"):
            if key not in segment:
                raise ValueError(f"line {index}, segment {seg_idx}: missing {key}")

        start = float(segment["start_time"])
        end = float(segment["end_time"])
        emotion = segment["emotion"]
        reason = segment["reason"]

        if end <= start:
            raise ValueError(f"line {index}, segment {seg_idx}: end_time <= start_time")
        if emotion not in ALLOWED_EMOTIONS:
            raise ValueError(f"line {index}, segment {seg_idx}: invalid emotion={emotion!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"line {index}, segment {seg_idx}: empty reason")

        if seg_idx == 0 and not close(start, 0.0):
            raise ValueError(f"line {index}: first segment must start at 0.0")
        if previous_end is not None and not close(start, previous_end):
            raise ValueError(
                f"line {index}, segment {seg_idx}: start_time {start} "
                f"does not match previous end_time {previous_end}"
            )
        if previous_emotion == emotion:
            raise ValueError(
                f"line {index}, segment {seg_idx}: adjacent segments share emotion {emotion!r}"
            )

        previous_end = end
        previous_emotion = emotion

    if not close(previous_end, duration):
        raise ValueError(
            f"line {index}: last end_time {previous_end} does not match duration {duration}"
        )


def main():
    parser = argparse.ArgumentParser(description="Validate DAR SFT JSON/JSONL annotations.")
    parser.add_argument("--dataset", default="dar_sft", help="Registered dataset name.")
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    configs = data_list([args.dataset])
    total = 0
    errors = []
    for config in configs:
        path = Path(config["annotation_path"])
        annotations = load_annotations(path)
        for offset, item in enumerate(annotations, start=1):
            total += 1
            try:
                validate_item(item, offset)
            except ValueError as exc:
                errors.append(f"{path}:{offset}: {exc}")
                if len(errors) >= args.max_errors:
                    break
        if len(errors) >= args.max_errors:
            break

    if errors:
        print("DAR SFT data validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"DAR SFT data validation passed: {total} examples.")


if __name__ == "__main__":
    main()
