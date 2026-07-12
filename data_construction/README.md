# DAR Camera-Ready Data Construction

This directory contains the code path aligned with the ECCV camera-ready paper.
The default runner is:

```bash
cd /path/to/DAR
GEMINI_API_KEY=... ./run_data_construction_pipeline.sh
```

For a small slice:

```bash
START_INDEX=0 END_INDEX=99 GEMINI_API_KEY=... ./run_data_construction_pipeline.sh
```

## Stage Mapping

| Stage | Script | Paper role | Output |
| --- | --- | --- | --- |
| 0 | `00_dar_preprocess_filter.py` | Duration filtering and static-frame removal | `00_preprocessed_manifest.jsonl` |
| 1a | `01_gemini2.5pro.py` | Gemini semantic event boundary proposals only | `01_gemini_semantic_events.jsonl` |
| 1b | `02_PySceneDetect.py` | Detect visual cuts and snap semantic boundaries within 0.5s | `02_pyscenedetect_aligned_events.jsonl` |
| 1c | `03_internvl35_event_integrity_refine.py` | InternVL3.5 event integrity verification and merge/split refinement | `03_internvl_refined_events.jsonl` |
| 2 | `04_qwen3vl_differential_description.py` | Incremental differential captioning without emotion labels | `04_qwen3vl_descriptions.jsonl` |
| 2b | `04b_grounding_dino_description_filter.py` | Optional Grounding DINO visual-faithfulness filter | `04b_grounded_descriptions.jsonl` |
| 3 | `05_qwen3vl_stream_affect_reason.py` | Stream-of-affect reasoning using DAR Top-3; four emotion-reason pairs | `05_qwen3vl_affect_reasoning.jsonl` |
| 4a | `06_qwen3omni_quality_judge.py` | Qwen3-Omni judge over five consistency dimensions | `06_qwen3omni_judge.jsonl` |
| 4b | `07_internvl35_quality_judge.py` | InternVL3.5 judge over the same five dimensions | `07_internvl35_judge.jsonl` |
| 4c | `08_dual_consistency_committee.py` | Dual-consistency aggregation and rewrite manifest | `08_committee_verified_annotations.jsonl` |

Compatibility entrypoints are kept for old script names:

- `geminifirst_desc.py` -> `04_qwen3vl_differential_description.py`
- `geminifirst_cot_2.py` -> `05_qwen3vl_stream_affect_reason.py`
- `inference.py` -> `06_qwen3omni_quality_judge.py`
- `inference_batch.py` -> `07_internvl35_quality_judge.py`

## Important Schema Choices

- Gemini event segmentation does not receive candidate emotions and does not output `emotion`.
- Qwen3-VL description does not receive candidate emotions and does not output `emotion`.
- DAR Top-3 emotions are first used by `05_qwen3vl_stream_affect_reason.py`.
- `05_qwen3vl_stream_affect_reason.py` outputs four ranked `emotion`-`reason` candidate pairs and a selected pair.
- If adjacent selected emotions are identical, the two segments are merged and the rationale is regenerated for the sustained affective phase.
- The judge scripts return actionable feedback over the five paper dimensions: visual grounding, causal logic, viewer centricity, temporal consistency, and answer consistency.

## Useful Environment Variables

| Variable | Meaning |
| --- | --- |
| `WORK_DIR` | Output directory, default `camera-ready/work` |
| `DATASET_ROOT` | DAR dataset root |
| `METADATA_JSON` | DAR metadata path |
| `LABELS_JSON` | DAR labels with Top-3 emotions |
| `SELECTED_IDS_JSON` | Optional JSON list of video ids |
| `START_INDEX`, `END_INDEX` | Slice the manifest for shard runs |
| `QWEN3VL_MODEL` | Qwen3-VL model path |
| `QWEN3_OMNI_MODEL` | Qwen3-Omni judge model path; defaults to `QWEN3VL_MODEL` if not set |
| `INTERNVL35_MODEL` | InternVL3.5 model path |
| `GROUNDING_DINO_MODEL` | Optional Grounding DINO model path |

Each runner step can be skipped with `RUN_<STAGE>=0`, e.g.:

```bash
RUN_PREPROCESS=0 RUN_GEMINI=0 ./run_data_construction_pipeline.sh
```
