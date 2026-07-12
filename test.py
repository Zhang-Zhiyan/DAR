#!/usr/bin/env python
"""
Inference and evaluation for DAR video emotion segmentation.
"""

import argparse
import json
import gc
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from tqdm import tqdm

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_QWEN_VL_UTILS = REPO_ROOT / "qwen-vl-utils" / "src"
if LOCAL_QWEN_VL_UTILS.exists():
    sys.path.insert(0, str(LOCAL_QWEN_VL_UTILS))
from qwen_vl_utils import process_vision_info

EMOTION_CANDIDATES = [
    "Awkwardness", "Empathic Pain", "Fear", "Anger", "Sadness", "Relief",
    "Boredom", "Joy", "Aesthetic Appreciation", "Adoration", "Admiration",
    "Amusement", "Satisfaction", "Disgust", "Sexual Desire", "Confusion",
    "Romance", "Craving", "Horror", "Excitement", "Nostalgia",
    "Awe (or Wonder)", "Interest", "Calmness", "Surprise", "Entrancement", "Anxiety"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/path/to/DAR-R1"))
    parser.add_argument("--test-jsonl", default=os.environ.get("TEST_JSONL", "/path/to/DAR/test.jsonl"))
    parser.add_argument("--output-jsonl", default=os.environ.get("OUT_JSONL", "/path/to/outputs/dar_r1_test_predictions.jsonl"))
    parser.add_argument("--video-root", default=os.environ.get("VIDEO_ROOT", "/path/to/DAR/videos"))
    parser.add_argument("--video-path-prefix", default=os.environ.get("VIDEO_PATH_PREFIX", ""))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "4")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "4096")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "16384")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.8")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1234")))
    return parser.parse_args()


def load_test_data(test_jsonl: str) -> List[Dict[str, Any]]:
    data = []
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                video_path = item.get("video", "")
                video_duration = item.get("video_duration")
                video_id = os.path.splitext(os.path.basename(video_path))[0]

                gt_segments = []
                conversations = item.get("conversations", [])
                if len(conversations) >= 2:
                    gpt_text = conversations[1].get("value", "")
                    try:
                        gt_parsed = json.loads(gpt_text)
                        gt_segments = gt_parsed.get("segments", [])
                    except json.JSONDecodeError:
                        pass

                data.append({
                    "video_id": video_id,
                    "video_path": video_path,
                    "video_duration": video_duration if video_duration else 30.0,
                    "gt_segments": gt_segments
                })
            except json.JSONDecodeError:
                continue
    return data


def resolve_video_path(video_path: str, video_root: str, video_path_prefix: str) -> str:
    if os.path.exists(video_path):
        return video_path
    if video_path_prefix and video_path.startswith(video_path_prefix):
        return os.path.join(video_root, video_path[len(video_path_prefix):])
    if not os.path.isabs(video_path):
        return os.path.join(video_root, video_path)
    return video_path


def load_done_ids(out_jsonl: str) -> Set[str]:
    done = set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                vid = str(j.get("video_id", ""))
                if vid and j.get("segments"):
                    done.add(vid)
            except Exception:
                continue
    return done


def build_eval_prompt(video_duration: float) -> str:
    cand_str = ", ".join(EMOTION_CANDIDATES)
    dur_str = f"{video_duration:.1f}" if video_duration and video_duration > 0 else "unknown"

    prompt = f"""You are analyzing a **SILENT video** (no audio/music/dialogue) to understand the **viewer's emotional journey**.

## Your Task
1. **Segment** the video into emotion phases based on when the viewer's dominant emotion CHANGES
2. **Predict** the viewer's emotion for each segment (must be from the allowed list)
3. **Reason** why the viewer feels that emotion

## Core Principles (Based on Appraisal-Event Theory)
- **Observation**: What visual elements do you see? (actions, objects, colors, lighting, composition)
- **Causality**: Why would these specific visual elements trigger a particular emotion in viewers?
- **Dynamics**: If the emotion changes from the previous segment, what specific event caused this transition?
- **Adaptive Merging**: Emotional responses persist until a new event triggers change. Therefore:
  - **CRITICAL: Adjacent segments MUST have DIFFERENT emotions**
  - If two consecutive time periods evoke the same emotion, they should be MERGED into ONE segment
  - Only create a new segment when the viewer's emotion genuinely CHANGES

## Viewer-Centered Focus
- This is about the **VIEWER's emotion** while watching, NOT the emotions of characters in the video
- Don't just describe what characters feel (e.g., "the man looks happy")
- Explain what makes the VIEWER feel a certain way

## Allowed Emotions (STRICTLY choose from these 27 only):
{cand_str}

## Hard Constraints
1. Segments MUST cover the full video continuously:
   - First segment starts at 0.0
   - Segments are continuous: segment[i].end_time == segment[i+1].start_time
   - Last segment ends at {dur_str} (video duration)
2. Timestamps: seconds with 1 decimal place
3. **CRITICAL: Adjacent segments MUST have DIFFERENT emotions** (merge if same)

## Quality Criteria for reason:
1. **Visual Grounding**: Reference visual cues that ACTUALLY exist in the video frames. Don't invent details.
2. **Causal Logic**: Your reasoning must logically connect visual evidence to emotion.
3. **Viewer-Centeredness**: Focus on the VIEWER's feelings, not just describing characters' facial expressions.
4. **Temporal Consistency**: Your reasoning should be consistent with the historical context (previous segments).

Output ONLY a JSON object with this schema:
{{
  "segments": [
    {{
      "start_time": 0.0,
      "end_time": <float>,
      "emotion": "<one of the 27 emotions>",
      "reason": "<viewer-centered, visually grounded reason>"
    }}, ...
  ]
}}"""
    return prompt


def make_messages(video_path: str, video_duration: float) -> List[Dict[str, Any]]:
    abs_vp = os.path.abspath(video_path)
    prompt_text = build_eval_prompt(video_duration)
    return [{
        "role": "user",
        "content": [
            {"type": "video", "video": abs_vp},
            {"type": "text", "text": prompt_text}
        ]
    }]


def build_input(processor, messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    return {'prompt': text, 'multi_modal_data': mm_data, 'mm_processor_kwargs': video_kwargs}


def try_parse_json(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline+1:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(s[l:r+1])
    except Exception:
        return None


def merge_adjacent_same_emotion_segments(segments: List[Dict]) -> List[Dict]:
    if not segments or len(segments) <= 1:
        return segments
    merged = []
    current = segments[0].copy()
    for i in range(1, len(segments)):
        next_seg = segments[i]
        if next_seg.get("emotion", "") == current.get("emotion", ""):
            current["end_time"] = next_seg.get("end_time", current["end_time"])
            cur_reason = current.get("reason", "")
            nxt_reason = next_seg.get("reason", "")
            if nxt_reason and nxt_reason not in cur_reason:
                current["reason"] = cur_reason + " " + nxt_reason
        else:
            merged.append(current)
            current = next_seg.copy()
    merged.append(current)
    return merged


def validate_and_fix_segments(segments: List[Dict], video_duration: float) -> List[Dict]:
    if not segments:
        return []
    segments.sort(key=lambda x: x.get("start_time", 0))
    for seg in segments:
        emo = seg.get("emotion", "")
        if emo not in EMOTION_CANDIDATES:
            matched = False
            for cand in EMOTION_CANDIDATES:
                if cand.lower() in emo.lower() or emo.lower() in cand.lower():
                    seg["emotion"] = cand
                    matched = True
                    break
            if not matched:
                seg["emotion"] = "Interest"
    segments = merge_adjacent_same_emotion_segments(segments)
    if segments[0].get("start_time", 0) != 0.0:
        segments[0]["start_time"] = 0.0
    if abs(segments[-1].get("end_time", 0) - video_duration) > 0.5:
        segments[-1]["end_time"] = video_duration
    for i in range(1, len(segments)):
        if segments[i]["start_time"] != segments[i-1]["end_time"]:
            segments[i]["start_time"] = segments[i-1]["end_time"]
    return segments


def calculate_iou(seg1: Dict, seg2: Dict) -> float:
    s1, e1 = seg1['start_time'], seg1['end_time']
    s2, e2 = seg2['start_time'], seg2['end_time']
    intersection = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return intersection / union if union > 0 else 0.0


def evaluate_single_video(pred_segments: List[Dict], gt_segments: List[Dict]) -> Dict:
    pred_count = len(pred_segments)
    gt_count = len(gt_segments)
    count_match = (pred_count == gt_count)

    compare_count = min(pred_count, gt_count)

    ious = []
    emotion_matches = []
    details = []

    for i in range(compare_count):
        pred_seg = pred_segments[i]
        gt_seg = gt_segments[i]

        iou = calculate_iou(pred_seg, gt_seg)
        ious.append(iou)

        pred_emo = pred_seg.get("emotion", "").strip()
        gt_emo = gt_seg.get("emotion", "").strip()
        emo_match = (pred_emo == gt_emo)
        emotion_matches.append(emo_match)

        details.append({
            "segment_idx": i,
            "pred_time": [pred_seg["start_time"], pred_seg["end_time"]],
            "gt_time": [gt_seg["start_time"], gt_seg["end_time"]],
            "iou": round(iou, 4),
            "pred_emotion": pred_emo,
            "gt_emotion": gt_emo,
            "emotion_match": emo_match
        })

    return {
        "count_match": count_match,
        "pred_count": pred_count,
        "gt_count": gt_count,
        "compared": compare_count,
        "avg_iou": sum(ious) / len(ious) if ious else 0.0,
        "emotion_accuracy": sum(emotion_matches) / len(emotion_matches) if emotion_matches else 0.0,
        "emotion_correct": sum(emotion_matches),
        "ious": ious,
        "emotion_matches": emotion_matches,
        "details": details
    }


def compute_overall_metrics(all_results: Dict[str, Dict]) -> Dict:
    total_videos = len(all_results)
    if total_videos == 0:
        return {}

    count_match_total = sum(1 for r in all_results.values() if r["count_match"])
    all_ious = []
    all_emotion_matches = []

    for r in all_results.values():
        all_ious.extend(r["ious"])
        all_emotion_matches.extend(r["emotion_matches"])

    return {
        "total_videos": total_videos,
        "segment_count_match": count_match_total,
        "segment_count_match_rate": count_match_total / total_videos,
        "total_compared_segments": len(all_ious),
        "avg_iou": sum(all_ious) / len(all_ious) if all_ious else 0.0,
        "emotion_accuracy": sum(all_emotion_matches) / len(all_emotion_matches) if all_emotion_matches else 0.0,
        "emotion_correct": sum(all_emotion_matches),
        "emotion_total": len(all_emotion_matches),
    }


if __name__ == "__main__":
    args = parse_args()

    print("=" * 80)
    print("DAR video emotion segmentation inference and evaluation")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Test JSONL: {args.test_jsonl}")
    print(f"Output JSONL: {args.output_jsonl}")
    print(f"Video root: {args.video_root}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    print("\nLoading test data...")
    test_data = load_test_data(args.test_jsonl)
    print(f"Loaded {len(test_data)} test videos")

    done_ids = load_done_ids(args.output_jsonl)
    print(f"Completed videos in existing output: {len(done_ids)}")
    videos_to_process = [v for v in test_data if v["video_id"] not in done_ids]
    print(f"Videos to process: {len(videos_to_process)}")

    if not videos_to_process:
        print("\nAll videos are already processed. Running evaluation.")
    else:
        print("\nInitializing model...")
        tensor_parallel_size = torch.cuda.device_count()
        print(f"GPU count: {tensor_parallel_size}")

        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=args.max_model_len,
            seed=args.seed,
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=args.max_tokens,
            stop_token_ids=[],
        )
        print("Model initialized")

        os.makedirs(os.path.dirname(args.output_jsonl) or '.', exist_ok=True)
        fout = open(args.output_jsonl, "a", encoding="utf-8")

        print("\nStarting inference...")
        success_count = 0
        fail_count = 0

        for batch_start in tqdm(range(0, len(videos_to_process), args.batch_size), desc="inference"):
            batch_videos = videos_to_process[batch_start:batch_start + args.batch_size]
            batch_inputs = []
            batch_meta = []

            for v in batch_videos:
                vid = v["video_id"]
                vp = resolve_video_path(v["video_path"], args.video_root, args.video_path_prefix)
                if not os.path.exists(vp):
                    result = {"video_id": vid, "video_path": vp, "segments": [], "error": "File not found"}
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    fail_count += 1
                    continue
                try:
                    messages = make_messages(vp, v["video_duration"])
                    inp = build_input(processor, messages)
                    batch_inputs.append(inp)
                    batch_meta.append(v)
                except Exception as e:
                    result = {"video_id": vid, "video_path": vp, "segments": [], "error": repr(e)}
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    fail_count += 1

            if not batch_inputs:
                continue

            try:
                outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
                for k, meta in enumerate(batch_meta):
                    vid = meta["video_id"]
                    duration = meta["video_duration"]
                    try:
                        raw_text = outputs[k].outputs[0].text
                        parsed = try_parse_json(raw_text)
                        if parsed and "segments" in parsed:
                            segments = validate_and_fix_segments(parsed["segments"], duration)
                            result = {"video_id": vid, "video_path": meta["video_path"],
                                      "video_duration": duration, "segments": segments}
                            success_count += 1
                        else:
                            result = {"video_id": vid, "video_path": meta["video_path"],
                                      "video_duration": duration, "segments": [],
                                      "error": "Parse failed", "raw_output": raw_text[:500]}
                            fail_count += 1
                    except Exception as e:
                        result = {"video_id": vid, "video_path": meta["video_path"],
                                  "segments": [], "error": repr(e)}
                        fail_count += 1
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                del outputs
                gc.collect()
            except Exception as e:
                for meta in batch_meta:
                    result = {"video_id": meta["video_id"], "segments": [], "error": repr(e)}
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    fail_count += 1

        fout.close()
        print(f"\nInference finished: success={success_count}, failed={fail_count}")

    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    gt_dict = {}
    for item in test_data:
        gt_dict[item["video_id"]] = item["gt_segments"]

    pred_dict = {}
    with open(args.output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                vid = j.get("video_id", "")
                segs = j.get("segments", [])
                if vid and segs:
                    pred_dict[vid] = segs
            except Exception:
                continue

    print(f"GT videos: {len(gt_dict)}")
    print(f"Predicted videos: {len(pred_dict)}")

    common_ids = set(gt_dict.keys()) & set(pred_dict.keys())
    print(f"Evaluable videos: {len(common_ids)}")

    all_eval_results = {}
    for vid in sorted(common_ids):
        result = evaluate_single_video(pred_dict[vid], gt_dict[vid])
        all_eval_results[vid] = result

    metrics = compute_overall_metrics(all_eval_results)

    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Evaluated videos:           {metrics.get('total_videos', 0)}")
    print(f"Segment count matches:      {metrics.get('segment_count_match', 0)}")
    print(f"Segment count match rate:   {metrics.get('segment_count_match_rate', 0):.4f}")
    print(f"Compared segments:          {metrics.get('total_compared_segments', 0)}")
    print(f"Mean temporal IoU:          {metrics.get('avg_iou', 0):.4f}")
    print(f"Emotion accuracy:           {metrics.get('emotion_accuracy', 0):.4f} ({metrics.get('emotion_correct', 0)}/{metrics.get('emotion_total', 0)})")
    print("=" * 80)

    print("\nEmotion accuracy under IoU thresholds:")
    all_ious = []
    all_emo_matches = []
    for r in all_eval_results.values():
        all_ious.extend(r["ious"])
        all_emo_matches.extend(r["emotion_matches"])

    for threshold in [0.3, 0.5, 0.7, 0.9]:
        valid = [(iou, em) for iou, em in zip(all_ious, all_emo_matches) if iou >= threshold]
        if valid:
            acc = sum(em for _, em in valid) / len(valid)
            print(f"  IOU >= {threshold:.1f}: {acc:.4f} ({sum(em for _, em in valid)}/{len(valid)} segments)")
        else:
            print(f"  IOU >= {threshold:.1f}: N/A (0 segments)")

    print("\nExample results (first 5 videos):")
    for vid in sorted(common_ids)[:5]:
        r = all_eval_results[vid]
        print(f"\n  video {vid}: pred={r['pred_count']} gt={r['gt_count']} "
              f"count_match={r['count_match']} "
              f"avgIoU={r['avg_iou']:.3f} emotion_acc={r['emotion_accuracy']:.3f}")
        for d in r["details"]:
            print(f"    seg{d['segment_idx']}: IOU={d['iou']:.3f} "
                  f"{d['pred_emotion']} vs {d['gt_emotion']} "
                  f"match={d['emotion_match']}")

    eval_output_path = args.output_jsonl.replace(".jsonl", "_metrics.json")
    eval_data = {
        "summary": metrics,
        "per_video": {vid: {k: v for k, v in r.items() if k != "details"} for vid, r in all_eval_results.items()}
    }
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics to: {eval_output_path}")
