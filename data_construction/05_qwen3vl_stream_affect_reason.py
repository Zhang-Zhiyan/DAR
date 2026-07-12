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
    DAR_EMOTIONS,
    append_jsonl,
    extract_json_object,
    iter_jsonl,
    load_done_video_ids,
    load_json,
    normalize_segments,
    resolve_video_path,
    round_time,
    top3_for_video,
)


DEFAULT_QWEN_UTILS = "/path/to/qwen-vl-utils/src"
if DEFAULT_QWEN_UTILS not in sys.path:
    sys.path.append(DEFAULT_QWEN_UTILS)
from qwen_vl_utils import process_vision_info  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL stream-of-affect reasoning for DAR")
    parser.add_argument("--model_path", default="/path/to/Qwen3-VL-32B-Instruct")
    parser.add_argument("--input_jsonl", default="/path/to/DAR/work/04b_grounded_descriptions.jsonl")
    parser.add_argument("--output_jsonl", default="/path/to/DAR/work/05_qwen3vl_affect_reasoning.jsonl")
    parser.add_argument("--dataset_root", default="/path/to/DAR/raw_dar")
    parser.add_argument("--metadata_json", default="/path/to/DAR/raw_dar/metadata.json")
    parser.add_argument("--labels_json", default="/path/to/DAR/raw_dar/all_labels.json")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    return parser.parse_args()


def top3_for_item(item: Dict[str, Any], metadata: Dict[str, Any], labels: Dict[str, Any]) -> List[str]:
    top3 = item.get("top3_emotions") or item.get("candidate_emotions") or []
    if isinstance(top3, list) and top3:
        return [str(x) for x in top3[:3]]
    return top3_for_video(str(item.get("video_id", "")), metadata, labels)


def previous_context_text(previous_segment: Optional[Dict[str, Any]]) -> str:
    if not previous_segment:
        return "Previous context: <empty>"
    return f"""Previous context:
S_(i-1): [{previous_segment['start_time']:.1f}s, {previous_segment['end_time']:.1f}s]
D_(i-1): {previous_segment.get('description', '')}
E_(i-1): {previous_segment.get('emotion', '')}
R_(i-1): {previous_segment.get('reason', '')}"""


def build_reason_prompt(
    video_id: str,
    segment: Dict[str, Any],
    segment_index: int,
    total_segments: int,
    top3_emotions: Sequence[str],
    previous_segment: Optional[Dict[str, Any]],
    merge_hint: str = "",
) -> str:
    st = float(segment["start_time"])
    et = float(segment["end_time"])
    event = str(segment.get("event", "")).strip()
    description = str(segment.get("description", "")).strip()
    top3_text = ", ".join(top3_emotions) if top3_emotions else "not available"
    taxonomy = ", ".join(DAR_EMOTIONS)

    return f"""You are generating viewer-centric dynamic affective reasoning for a silent video.

Use the Bottom-up reasoning logic:
Visual Evidence -> Contextual Appraisal -> Emotion Trigger.

Important:
- Emotion means the viewer's induced emotion, not a character's emotion.
- Ground every reason in visible evidence from the target segment.
- Consider the previous segment context to explain transitions or sustained affect.
- Do not mention audio, sound, music, speech, or dialogue.

Video id: {video_id}
Current segment: {segment_index + 1}/{total_segments}
S_i: [{st:.1f}s, {et:.1f}s]
Event cue: {event}
D_i: {description}

{previous_context_text(previous_segment)}

DAR Top-3 candidate emotions for this video:
[{top3_text}]

Full DAR emotion taxonomy:
[{taxonomy}]

{merge_hint}

Task:
Generate exactly four candidate emotion-reason pairs for S_i. The four pairs
should be ranked from most plausible to least plausible. Prioritize the Top-3
DAR candidates, but you may use another taxonomy emotion if the visual evidence
strongly supports it. Then choose the best candidate as the selected emotion.

Return ONLY JSON:
{{
  "candidate_pairs": [
    {{"emotion": "<emotion label>", "reason": "<80-160 words grounded causal rationale>"}},
    {{"emotion": "<emotion label>", "reason": "<80-160 words grounded causal rationale>"}},
    {{"emotion": "<emotion label>", "reason": "<80-160 words grounded causal rationale>"}},
    {{"emotion": "<emotion label>", "reason": "<80-160 words grounded causal rationale>"}}
  ],
  "selected_pair_index": 0,
  "selected_emotion": "<emotion label>",
  "selected_reason": "<the selected reason>"
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


def normalize_reason_output(parsed: Optional[Dict[str, Any]], raw_text: str) -> Dict[str, Any]:
    if not parsed:
        return {
            "candidate_pairs": [{"emotion": "", "reason": raw_text.strip()}],
            "selected_pair_index": 0,
            "selected_emotion": "",
            "selected_reason": raw_text.strip(),
            "parse_error": True,
        }

    pairs = parsed.get("candidate_pairs") or parsed.get("pairs") or []
    if not isinstance(pairs, list):
        pairs = []
    clean_pairs = []
    for pair in pairs[:4]:
        if not isinstance(pair, dict):
            continue
        clean_pairs.append(
            {
                "emotion": str(pair.get("emotion", "")).strip(),
                "reason": str(pair.get("reason", "")).strip(),
            }
        )
    while len(clean_pairs) < 4:
        clean_pairs.append({"emotion": "", "reason": ""})

    selected_index = parsed.get("selected_pair_index", 0)
    try:
        selected_index = int(selected_index)
    except Exception:
        selected_index = 0
    selected_index = min(max(selected_index, 0), 3)

    selected_emotion = str(parsed.get("selected_emotion") or clean_pairs[selected_index]["emotion"]).strip()
    selected_reason = str(parsed.get("selected_reason") or clean_pairs[selected_index]["reason"]).strip()
    return {
        "candidate_pairs": clean_pairs,
        "selected_pair_index": selected_index,
        "selected_emotion": selected_emotion,
        "selected_reason": selected_reason,
    }


def generate_reason(
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    video_path: str,
    video_id: str,
    segment: Dict[str, Any],
    segment_index: int,
    total_segments: int,
    top3_emotions: Sequence[str],
    previous_segment: Optional[Dict[str, Any]],
    merge_hint: str = "",
) -> Dict[str, Any]:
    prompt = build_reason_prompt(
        video_id,
        segment,
        segment_index,
        total_segments,
        top3_emotions,
        previous_segment,
        merge_hint=merge_hint,
    )
    messages = make_messages(video_path, segment, prompt)
    model_input = build_input(processor, messages)
    output = llm.generate([model_input], sampling_params=sampling_params)[0]
    raw_text = output.outputs[0].text
    parsed = extract_json_object(raw_text)
    result = normalize_reason_output(parsed, raw_text)
    result["raw_text"] = raw_text[:1000] if result.get("parse_error") else ""
    return result


def segment_with_reason(segment: Dict[str, Any], reason_result: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "start_time": round_time(segment["start_time"]),
        "end_time": round_time(segment["end_time"]),
        "event": segment.get("event", ""),
        "description": segment.get("description", ""),
        "emotion": reason_result.get("selected_emotion", ""),
        "reason": reason_result.get("selected_reason", ""),
        "candidate_pairs": reason_result.get("candidate_pairs", []),
        "selected_pair_index": reason_result.get("selected_pair_index", 0),
    }
    if reason_result.get("parse_error"):
        out["parse_error"] = True
        out["raw_text"] = reason_result.get("raw_text", "")
    return out


def merge_segments(previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start_time": previous["start_time"],
        "end_time": current["end_time"],
        "event": f"{previous.get('event', '')} Then, {current.get('event', '')}".strip(),
        "description": f"{previous.get('description', '')}\nThen, {current.get('description', '')}".strip(),
    }


def reason_video(
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    item: Dict[str, Any],
    top3_emotions: Sequence[str],
) -> Dict[str, Any]:
    video_id = str(item["video_id"])
    video_path = item["video_path"]
    segments = normalize_segments(item["segments"], float(item["video_duration"]), keep_keys=("event", "description"))

    final_segments: List[Dict[str, Any]] = []
    merge_log: List[Dict[str, Any]] = []

    for original_idx, segment in enumerate(segments):
        previous = final_segments[-1] if final_segments else None
        result = generate_reason(
            llm,
            processor,
            sampling_params,
            video_path,
            video_id,
            segment,
            original_idx,
            len(segments),
            top3_emotions,
            previous,
        )
        current = segment_with_reason(segment, result)

        if previous and current.get("emotion") and current.get("emotion") == previous.get("emotion"):
            popped = final_segments.pop()
            merged = merge_segments(popped, current)
            merged_previous = final_segments[-1] if final_segments else None
            merge_hint = (
                f"The previous and current segments were both predicted as {current['emotion']}. "
                "Regenerate the emotion-reason pairs for the merged interval as one sustained affective phase."
            )
            merged_result = generate_reason(
                llm,
                processor,
                sampling_params,
                video_path,
                video_id,
                merged,
                max(0, len(final_segments)),
                len(segments),
                top3_emotions,
                merged_previous,
                merge_hint=merge_hint,
            )
            merged_out = segment_with_reason(merged, merged_result)
            final_segments.append(merged_out)
            merge_log.append(
                {
                    "merged_original_segment_index": original_idx,
                    "emotion": current["emotion"],
                    "start_time": merged_out["start_time"],
                    "end_time": merged_out["end_time"],
                }
            )
        else:
            final_segments.append(current)
        gc.collect()

    out = {k: v for k, v in item.items() if k not in ("segments",)}
    out["stage"] = "qwen3vl_stream_of_affect_reasoning"
    out["top3_emotions"] = list(top3_emotions)
    out["segments"] = final_segments
    out["merge_log"] = merge_log
    return out


def main() -> None:
    args = parse_args()
    metadata = load_json(args.metadata_json) if args.metadata_json and Path(args.metadata_json).exists() else {}
    labels = load_json(args.labels_json) if args.labels_json and Path(args.labels_json).exists() else {}

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

    print(f"Qwen3-VL affect reasoning: pending={len(pending)}, done={len(done)}")
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
        top_p=0.9,
        top_k=20,
        max_tokens=args.max_tokens,
        stop_token_ids=[],
    )

    for item in tqdm(pending, desc="Reasoning"):
        try:
            top3 = top3_for_item(item, metadata, labels)
            if not top3:
                top3 = DAR_EMOTIONS[:3]
            record = reason_video(llm, processor, sampling_params, item, top3)
        except Exception as exc:
            record = {
                "video_id": item.get("video_id"),
                "video_path": item.get("video_path"),
                "video_duration": item.get("video_duration"),
                "top3_emotions": item.get("top3_emotions", []),
                "error": f"reasoning_failed: {exc}",
            }
        append_jsonl(args.output_jsonl, record)

    print(f"Done. Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
