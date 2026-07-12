#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DAR_EMOTIONS = [
    "Admiration",
    "Amusement",
    "Anger",
    "Anxiety",
    "Awe",
    "Awe (or Wonder)",
    "Boredom",
    "Calmness",
    "Confusion",
    "Craving",
    "Disgust",
    "Empathic Pain",
    "Entrancement",
    "Excitement",
    "Fear",
    "Horror",
    "Interest",
    "Joy",
    "Nostalgia",
    "Relief",
    "Romance",
    "Sadness",
    "Satisfaction",
    "Sexual Desire",
    "Surprise",
    "Awkwardness",
    "Adoration",
    "Aesthetic Appreciation",
]


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str | os.PathLike[str]) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def append_jsonl(path: str | os.PathLike[str], record: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path: str | os.PathLike[str], records: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_video_ids(path: str | os.PathLike[str]) -> set[str]:
    if not Path(path).exists():
        return set()
    done = set()
    for item in iter_jsonl(path):
        vid = item.get("video_id")
        if vid is not None:
            done.add(str(vid))
    return done


def round_time(value: Any, ndigits: int = 1) -> float:
    try:
        return round(float(value), ndigits)
    except Exception:
        return 0.0


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None

    if s.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
        if match:
            s = match.group(1).strip()

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    left = s.find("{")
    right = s.rfind("}")
    if left == -1 or right <= left:
        return None
    try:
        obj = json.loads(s[left : right + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def resolve_video_path(video_path: str, dataset_root: str, video_id: str = "") -> str:
    if video_path:
        p = Path(video_path)
        if p.is_absolute() and p.exists():
            return str(p)
        candidate = Path(dataset_root) / video_path
        if candidate.exists():
            return str(candidate)
        candidate = Path(dataset_root) / "videos" / p.name
        if candidate.exists():
            return str(candidate)

    if video_id:
        for candidate in (
            Path(dataset_root) / "videos" / f"{video_id}.mp4",
            Path(dataset_root) / f"{video_id}.mp4",
        ):
            if candidate.exists():
                return str(candidate)
    return video_path


def duration_from_metadata(metadata: Dict[str, Any], video_id: str) -> Optional[float]:
    item = metadata.get(str(video_id), {})
    duration = item.get("duration")
    if duration is None:
        return None
    try:
        return float(duration)
    except Exception:
        return None


def file_from_metadata(metadata: Dict[str, Any], video_id: str) -> str:
    item = metadata.get(str(video_id), {})
    return str(item.get("file") or item.get("video_path") or "")


def top3_from_record(record: Dict[str, Any]) -> List[str]:
    for key in ("top3_emotions", "top3emotion", "candidate_emotions", "topK", "topk"):
        value = record.get(key)
        if isinstance(value, list):
            return [str(x) for x in value[:3] if str(x).strip()]
    emotions = record.get("emotions")
    if isinstance(emotions, dict):
        ranked = sorted(emotions.items(), key=lambda kv: float(kv[1] or 0), reverse=True)
        return [str(k) for k, _ in ranked[:3]]
    return []


def top3_for_video(
    video_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if labels and str(video_id) in labels:
        top3 = top3_from_record(labels[str(video_id)])
        if top3:
            return top3
    if metadata and str(video_id) in metadata:
        top3 = top3_from_record(metadata[str(video_id)])
        if top3:
            return top3
    return []


def normalize_segments(
    segments: Sequence[Dict[str, Any]],
    duration: float,
    min_duration: float = 0.1,
    keep_keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    dur = round_time(duration)
    if dur <= 0:
        return []

    clean: List[Dict[str, Any]] = []
    for seg in segments:
        try:
            start = float(seg.get("start_time", seg.get("start", 0.0)))
            end = float(seg.get("end_time", seg.get("end", 0.0)))
        except Exception:
            continue
        if end <= start:
            continue
        item = dict(seg)
        item["start_time"] = clamp(start, 0.0, dur)
        item["end_time"] = clamp(end, 0.0, dur)
        clean.append(item)

    if not clean:
        return [{"start_time": 0.0, "end_time": dur, "event": "Full video event"}]

    clean.sort(key=lambda x: (float(x["start_time"]), float(x["end_time"])))
    fixed: List[Dict[str, Any]] = []
    prev_end = 0.0
    n = len(clean)
    for idx, seg in enumerate(clean):
        start = 0.0 if idx == 0 else prev_end
        if idx == n - 1:
            end = dur
        else:
            remaining = n - idx - 1
            max_end = round_time(dur - remaining * min_duration)
            end = round_time(seg["end_time"])
            end = clamp(end, start + min_duration, max_end)
        out = {}
        if keep_keys:
            for key in keep_keys:
                if key in seg:
                    out[key] = seg[key]
        else:
            out.update(seg)
        out["start_time"] = round_time(start)
        out["end_time"] = round_time(end)
        if "event" in out:
            out["event"] = str(out.get("event") or "").strip() or "Visual event"
        fixed.append(out)
        prev_end = out["end_time"]
    fixed[-1]["end_time"] = dur
    return fixed


def nearest_cut(boundary: float, cuts: Sequence[float], window: float) -> Tuple[Optional[float], Optional[float]]:
    if not cuts:
        return None, None
    best = min(cuts, key=lambda cut: abs(cut - boundary))
    delta = abs(best - boundary)
    if delta <= window:
        return round_time(best), round_time(delta)
    return None, None


def get_video_id_from_path(path: str) -> str:
    return Path(path).stem
