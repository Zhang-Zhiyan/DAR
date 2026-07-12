from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from swift.plugin import ORM, orms


EMOTION_LIST = [
    "Awkwardness", "Empathic Pain", "Fear", "Anger", "Sadness", "Relief",
    "Boredom", "Joy", "Aesthetic Appreciation", "Adoration", "Admiration",
    "Amusement", "Satisfaction", "Disgust", "Sexual Desire", "Confusion",
    "Romance", "Craving", "Horror", "Excitement", "Nostalgia",
    "Awe (or Wonder)", "Interest", "Calmness", "Surprise", "Entrancement",
    "Anxiety",
]
EMOTION_SET = set(EMOTION_LIST)

TAU_MIN_SEGMENT = 0.4
LEN_SOFT = 1200
LEN_HARD = 2400
LEN_TAU = 400.0
TAU_COUNT = 1.5
TAU_TEMP = 0.5
SEG_ALPHA = 0.5
REASON_MU = 120
REASON_SIGMA = 40
REASON_DUP_TAU = 0.75


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return content if isinstance(content, str) else json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts) if parts else str(value)
    return "" if value is None else str(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _round1(value: float) -> float:
    return round(float(value), 1)


def _is_one_decimal(value: float) -> bool:
    return abs(float(value) - round(float(value), 1)) < 1e-9


def _strip_code_fences(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _extract_json_object(text: str) -> Optional[str]:
    value = (text or "").strip()
    left = value.find("{")
    right = value.rfind("}")
    if left == -1 or right == -1 or right <= left:
        return None
    return value[left:right + 1]


def _load_json_object(text: str, *, strict: bool) -> Optional[Dict[str, Any]]:
    value = _strip_code_fences(_as_text(text))
    if not strict:
        value = _extract_json_object(value) or value
    try:
        obj = json.loads(value)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _segment_value(segment: Dict[str, Any], primary: str, fallback: str) -> Any:
    return segment.get(primary) if primary in segment else segment.get(fallback)


def _normalize_segments(raw_segments: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_segments, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        start = _safe_float(_segment_value(segment, "start_time", "start"))
        end = _safe_float(_segment_value(segment, "end_time", "end"))
        if start is None or end is None:
            continue
        emotion = segment.get("emotion")
        reason = segment.get("reason", "")
        normalized.append({
            "start": start,
            "end": end,
            "emotion": str(emotion) if emotion is not None else "",
            "reason": str(reason) if reason is not None else "",
        })
    return normalized


def _parse_pred_segments(text: Any, *, strict: bool = False) -> List[Dict[str, Any]]:
    obj = _load_json_object(_as_text(text), strict=strict)
    if obj is None:
        return []
    return _normalize_segments(obj.get("segments"))


def _parse_gt_segments(solution: Any) -> List[Dict[str, Any]]:
    if isinstance(solution, dict):
        obj = solution
    else:
        obj = _load_json_object(_as_text(solution), strict=False)
    if obj is None:
        return []
    return _normalize_segments(obj.get("segments"))


def _solution_at(solution: Optional[Sequence[Any]], index: int) -> Any:
    if solution is None:
        return None
    if index < len(solution):
        return solution[index]
    return None


def _interval_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return 0.0 if union <= 0 else inter / union


def _interval_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def _best_match_by_iou(pred_segment: Dict[str, Any], gt_segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not gt_segments:
        return None
    return max(gt_segments, key=lambda gt: _interval_iou(pred_segment, gt))


def _best_match_by_overlap(pred_segment: Dict[str, Any], gt_segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not gt_segments:
        return None
    return max(gt_segments, key=lambda gt: _interval_overlap(pred_segment, gt))


def _word_count(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text or "")
    if tokens:
        return len(tokens)
    stripped = (text or "").strip()
    return max(0, len(stripped) // 2)


def _token_set(text: str) -> set:
    return set(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", (text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    set_a = _token_set(a)
    set_b = _token_set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _length_score(text: str) -> float:
    n = _word_count(text)
    if n >= LEN_HARD:
        return 0.0
    return float(math.exp(-max(0, n - LEN_SOFT) / LEN_TAU))


def _schema_and_time_valid(obj: Optional[Dict[str, Any]], gt_segments: List[Dict[str, Any]]) -> bool:
    if obj is None or set(obj.keys()) != {"segments"}:
        return False
    segments = obj.get("segments")
    if not isinstance(segments, list) or not segments:
        return False

    expected_duration = gt_segments[-1]["end"] if gt_segments else None
    previous_end = None
    previous_emotion = None

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            return False
        required = {"start_time", "end_time", "emotion", "reason"}
        if not required.issubset(segment.keys()):
            return False

        start = _safe_float(segment.get("start_time"))
        end = _safe_float(segment.get("end_time"))
        if start is None or end is None:
            return False
        if not (_is_one_decimal(start) and _is_one_decimal(end)):
            return False
        if end <= start:
            return False
        if index == 0 and abs(_round1(start) - 0.0) > 1e-9:
            return False
        if previous_end is not None and abs(_round1(start) - _round1(previous_end)) > 1e-9:
            return False

        emotion = str(segment.get("emotion"))
        if emotion not in EMOTION_SET:
            return False
        if previous_emotion == emotion:
            return False
        if not isinstance(segment.get("reason"), str) or not segment["reason"].strip():
            return False

        previous_end = end
        previous_emotion = emotion

    if expected_duration is not None and abs(_round1(previous_end) - _round1(expected_duration)) > 1e-9:
        return False
    return True


def _debug(tag: str, completions: Sequence[Any], solution: Optional[Sequence[Any]], **kwargs) -> None:
    if os.environ.get("DAR_REWARD_DEBUG", "0") != "1":
        return
    trainer_state = kwargs.get("trainer_state")
    step = getattr(trainer_state, "global_step", None) if trainer_state is not None else None
    every = int(os.environ.get("DAR_REWARD_DEBUG_EVERY", "50"))
    if step is not None and every > 0 and step % every != 0:
        return
    print(f"[DAR-REWARD-DEBUG] step={step} tag={tag}")
    if completions:
        print(_as_text(completions[0])[:800])
    if solution:
        print(_as_text(solution[0])[:800])


class DARStructuralReward(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        _debug("struct", completions, solution, **kwargs)
        rewards: List[float] = []
        for index, completion in enumerate(completions):
            text = _as_text(completion)
            gt = _parse_gt_segments(_solution_at(solution, index))
            strict_obj = _load_json_object(text, strict=True)
            loose_pred = _parse_pred_segments(text, strict=False)

            r_fmt = 1.0 if _schema_and_time_valid(strict_obj, gt) else 0.0
            r_dur = (
                1.0
                if loose_pred and all((segment["end"] - segment["start"]) >= TAU_MIN_SEGMENT for segment in loose_pred)
                else 0.0
            )
            r_len = _length_score(text)
            reward = 0.5 * r_fmt + 0.2 * r_dur + 0.3 * r_len
            rewards.append(float(max(0.0, min(1.0, reward))))
        return rewards


class DARSegmentCountReward(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        _debug("count", completions, solution, **kwargs)
        rewards: List[float] = []
        for index, completion in enumerate(completions):
            pred = _parse_pred_segments(completion, strict=False)
            gt = _parse_gt_segments(_solution_at(solution, index))
            if not pred or not gt:
                rewards.append(0.0)
                continue
            reward = math.exp(-abs(len(pred) - len(gt)) / TAU_COUNT)
            rewards.append(float(max(0.0, min(1.0, reward))))
        return rewards


class DARTemporalSegmentationReward(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        _debug("seg", completions, solution, **kwargs)
        rewards: List[float] = []
        for index, completion in enumerate(completions):
            pred = _parse_pred_segments(completion, strict=False)
            gt = _parse_gt_segments(_solution_at(solution, index))
            if not pred or not gt:
                rewards.append(0.0)
                continue

            per_segment = []
            for pred_segment in pred:
                matched = _best_match_by_iou(pred_segment, gt)
                if matched is None:
                    continue
                iou = _interval_iou(pred_segment, matched)
                boundary_dist = (
                    abs(pred_segment["start"] - matched["start"])
                    + abs(pred_segment["end"] - matched["end"])
                )
                boundary_score = math.exp(-boundary_dist / TAU_TEMP)
                per_segment.append(SEG_ALPHA * iou + (1.0 - SEG_ALPHA) * boundary_score)

            reward = sum(per_segment) / len(pred) if per_segment else 0.0
            rewards.append(float(max(0.0, min(1.0, reward))))
        return rewards


class DAREmotionAccuracyReward(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        _debug("emo", completions, solution, **kwargs)
        rewards: List[float] = []
        for index, completion in enumerate(completions):
            pred = _parse_pred_segments(completion, strict=False)
            gt = _parse_gt_segments(_solution_at(solution, index))
            if not pred or not gt:
                rewards.append(0.0)
                continue

            scores = []
            for pred_segment in pred:
                matched = _best_match_by_overlap(pred_segment, gt)
                if matched is None:
                    scores.append(0.0)
                    continue
                ok = (
                    pred_segment["emotion"] == matched["emotion"]
                    and _interval_iou(pred_segment, matched) > 0.5
                )
                scores.append(1.0 if ok else 0.0)
            reward = sum(scores) / len(pred)
            rewards.append(float(max(0.0, min(1.0, reward))))
        return rewards


class DARReasoningQualityReward(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        _debug("reason", completions, solution, **kwargs)
        rewards: List[float] = []
        for completion in completions:
            pred = _parse_pred_segments(completion, strict=False)
            if not pred:
                rewards.append(0.0)
                continue

            scores = []
            previous_reason = ""
            for idx, segment in enumerate(pred):
                reason = segment.get("reason", "")
                if not reason.strip():
                    scores.append(0.0)
                    previous_reason = reason
                    continue
                length_term = math.exp(-abs(_word_count(reason) - REASON_MU) / REASON_SIGMA)
                duplicate = idx > 0 and _jaccard(reason, previous_reason) > REASON_DUP_TAU
                scores.append(length_term * (0.0 if duplicate else 1.0))
                previous_reason = reason

            reward = sum(scores) / len(pred)
            rewards.append(float(max(0.0, min(1.0, reward))))
        return rewards


orms["dar_struct"] = DARStructuralReward
orms["dar_count"] = DARSegmentCountReward
orms["dar_seg"] = DARTemporalSegmentationReward
orms["dar_emo"] = DAREmotionAccuracyReward
orms["dar_reason"] = DARReasoningQualityReward

orms["dar_format"] = DARStructuralReward
orms["dar_count"] = DARSegmentCountReward
orms["dar_segment"] = DARTemporalSegmentationReward
orms["dar_emotion"] = DAREmotionAccuracyReward
orms["dar_reason"] = DARReasoningQualityReward
orms["dar_min_dur"] = DARStructuralReward
orms["dar_len_penalty"] = DARStructuralReward
