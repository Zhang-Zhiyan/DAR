#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_solution(item: Dict[str, Any]) -> str:
    conversations = item.get("conversations", [])
    for turn in reversed(conversations):
        if turn.get("from") == "gpt":
            value = turn.get("value", "")
            json.loads(value)
            return value
    raise ValueError("missing gpt solution turn")


def convert_item(item: Dict[str, Any], index: int, system_prompt: str) -> Dict[str, Any]:
    video = item.get("video") or item.get("video_path")
    if not video:
        raise ValueError(f"item {index}: missing video path")

    duration = item.get("video_duration")
    if duration is None:
        solution_obj = json.loads(extract_solution(item))
        segments = solution_obj.get("segments") or []
        duration = segments[-1]["end_time"] if segments else "unknown"
    duration_text = f"{float(duration):.1f}s" if isinstance(duration, (int, float)) else f"{duration}"

    return {
        "id": item.get("id") or f"dar-grpo-{index:06d}",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "<video>\n"
                    f"Analyze this SILENT video (duration: {duration_text}).\n"
                    "Segment by when the viewer's dominant emotion changes; "
                    "choose an allowed emotion for each segment and explain why "
                    "based on the visuals."
                ),
            },
        ],
        "videos": [video],
        "solution": extract_solution(item),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/path/to/DAR/train.jsonl",
        help="DAR camera-ready SFT JSONL.",
    )
    parser.add_argument(
        "--output",
        default="/path/to/DAR/train_qwen25vl_ms_grpo.jsonl",
        help="Output JSONL consumed by ms-swift GRPO.",
    )
    parser.add_argument(
        "--prompt",
        default=str(Path(__file__).resolve().parents[1] / "prompt.txt"),
        help="System prompt text file.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    prompt_path = Path(args.prompt)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for index, item in enumerate(read_jsonl(input_path)):
            if args.max_samples is not None and written >= args.max_samples:
                break
            converted = convert_item(item, index, system_prompt)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} GRPO examples to {output_path}")


if __name__ == "__main__":
    main()
