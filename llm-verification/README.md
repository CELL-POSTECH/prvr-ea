# Dense Multi-GT Labeling

This directory contains the LLM verification pipeline used to build dense
multi-GT labels for PRVR evaluation.

Supported datasets:

```text
<activitynet|tvr|charades>
```

Run commands from the repository root unless noted.

Released labels are packaged at `datasets/multigt_labels.zip`:

```bash
unzip -q datasets/multigt_labels.zip -d datasets/
```

## Environment

PRVR candidate mining uses the repository training/evaluation environment:

```bash
bash env.sh
conda activate prvr
export PRVR_DATA_ROOT=/path/to/datasets
```

Qwen verification uses a separate environment:

```bash
bash llm-verification/setup_qwen3vl.sh
conda activate qwen3vl
```

Dependency specs:

```text
requirements-qwen3vl.txt
environment-qwen3vl.yml
```

## Expected Paths

Use paths that match your machine.

```text
datasets/<dataset>/
outputs/upstream/
outputs/runs/<dataset>/
outputs/<dataset>_rankings_from_ckpts/
all_prvr/<model>/results/
```

Candidate mining writes:

```text
outputs/upstream/pseudo_gt_candidates.<dataset>.jsonl
outputs/<dataset>_rankings_from_ckpts/<method>_top100.jsonl
outputs/<dataset>_rankings_from_ckpts/pure_clip_frame_topP1000_top100.jsonl
```

Verification writes:

```text
outputs/runs/<dataset>/verification.jsonl
outputs/runs/<dataset>/summary.json
outputs/runs/<dataset>/extra_gt_all.jsonl
```

## 1. Mine Candidates

Train four PRVR models in your environment, then pass their checkpoint paths to
the candidate miner. The miner also runs CLIP frame-level retrieval.

For the local batch script, edit checkpoint paths at the top of
`scripts/run_candidate_mining.sh`, then run:

```bash
bash scripts/run_candidate_mining.sh all --gpu 0
```

Run one dataset only:

```bash
bash scripts/run_candidate_mining.sh activitynet --gpu 0
```

Manual command format:

```bash
PRVR_DATA_ROOT=<data_root> python scripts/build_pseudo_gt_from_ckpts.py \
  --dataset activitynet \
  --prvr-model DreamPRVR=dreamprvr=<repo_root>/all_prvr/CVPR26-DreamPRVR/results/clip/activitynet/DreamPRVR/best.ckpt \
  --prvr-model GMMFormerv2=gmmformer=<repo_root>/all_prvr/GMMFormer_v2/results/clip/activitynet/gmmformer_v2/best.ckpt \
  --prvr-model HLFormer=hlformer=<repo_root>/all_prvr/ICCV25-HLFormer/results/clip/activitynet/HLFormer/best.ckpt \
  --prvr-model Holmes=holmes=<repo_root>/all_prvr/ICML26-Holmes/results/clip/activitynet/Holmes/<run>/best.ckpt \
  --output outputs/upstream/pseudo_gt_candidates.activitynet.jsonl \
  --gpu 0
```

`<adapter>` selects the model-code adapter used to run the checkpoint:

```text
dreamprvr
ms-sl
gmmformer
hlformer
holmes
```

The script writes per-method top-k rankings, computes pure CLIP frame-level
top-k, and keeps videos that appear in all PRVR rankings and pure CLIP top-k.
Default `--topk` is `100`.

The output is JSONL, one candidate pair per line:

```json
{
  "dataset": "activitynet",
  "query_key": "v_uqiMw7tQ1Cc#enc#0",
  "desc_id": "0",
  "query": "A weight lifting tutorial is given.",
  "type": "v",
  "original_gt_video_id": "v_uqiMw7tQ1Cc",
  "pseudo_video_id": "v_-01K1HxqPB8",
  "gt_ts": [0.28, 55.15],
  "gt_duration": 55.15,
  "pseudo_clip_feature_index": 286,
  "pseudo_clip_feature_len": 404,
  "gt_frame_times": [0.28, 8.118571428571428, 15.957142857142856, 23.795714285714286, 31.634285714285713, 39.472857142857144, 47.31142857142857, 55.15],
  "gt_frame_paths": [
    "<data_root>/activitynet/raw_frames/v_uqiMw7tQ1Cc/0000000280.jpg",
    "<data_root>/activitynet/raw_frames/v_uqiMw7tQ1Cc/0000008119.jpg",
    "<data_root>/activitynet/raw_frames/v_uqiMw7tQ1Cc/0000015957.jpg",
    "<data_root>/activitynet/raw_frames/v_uqiMw7tQ1Cc/0000023796.jpg",
    "<data_root>/activitynet/raw_frames/v_uqiMw7tQ1Cc/0000031634.jpg",
    "<data_root>/activitynet/raw_frames/v_uqiMw7tQ1Cc/0000039473.jpg",
    "<data_root>/activitynet/raw_frames/v_uqiMw7tQ1Cc/0000047311.jpg",
    "<data_root>/activitynet/raw_frames/v_uqiMw7tQ1Cc/0000055150.jpg"
  ],
  "pseudo_center_time": 153.08451612903227,
  "pseudo_window_mode": "gt_clamped",
  "pseudo_window_sec": 20.0,
  "pseudo_frame_times": [143.08451612903227, 145.94165898617513, 148.798801843318, 151.65594470046085, 154.51308755760368, 157.37023041474654, 160.2273732718894, 163.08451612903227],
  "pseudo_frame_paths": [
    "<data_root>/activitynet/raw_frames/v_-01K1HxqPB8/0000143085.jpg",
    "<data_root>/activitynet/raw_frames/v_-01K1HxqPB8/0000145942.jpg",
    "<data_root>/activitynet/raw_frames/v_-01K1HxqPB8/0000148799.jpg",
    "<data_root>/activitynet/raw_frames/v_-01K1HxqPB8/0000151656.jpg",
    "<data_root>/activitynet/raw_frames/v_-01K1HxqPB8/0000154513.jpg",
    "<data_root>/activitynet/raw_frames/v_-01K1HxqPB8/0000157370.jpg",
    "<data_root>/activitynet/raw_frames/v_-01K1HxqPB8/0000160227.jpg",
    "<data_root>/activitynet/raw_frames/v_-01K1HxqPB8/0000163085.jpg"
  ],
  "model_agreement": {
    "DreamPRVR": {"rank": 66, "score": 0.3960685133934021},
    "GMMFormerv2": {"rank": 75, "score": 0.3711954951286316},
    "HLFormer": {"rank": 60, "score": 0.3807704746723175},
    "Holmes": {"rank": 58, "score": 0.4048531651496887},
    "pure_clip": {"rank": 52, "score": 0.3037784993648529, "clip_feature_index": 286}
  }
}
```

Use `extract_query_candidates.py` only to create a small subset from an existing
candidate JSONL:

```bash
DATASET=<activitynet|tvr|charades>

python llm-verification/extract_query_candidates.py \
  --dataset "$DATASET" \
  --input outputs/upstream/pseudo_gt_candidates."$DATASET".jsonl \
  --query-key <query_id> \
  --output outputs/runs/query_subset/candidates.jsonl
```

## 2. Prepare Frames

Check frame paths before running Qwen:

```bash
DATASET=<activitynet|tvr|charades>

python llm-verification/sanity_check_candidate_frames.py \
  --dataset "$DATASET" \
  --input outputs/upstream/pseudo_gt_candidates."$DATASET".jsonl \
  --caption-file datasets/"$DATASET"/TextData/"$DATASET"val.caption.txt \
  --workers 32 \
  --fail-on-issue
```

If candidate rows contain frame timestamps instead of existing images,
materialize them from raw videos:

```bash
DATASET=<activitynet|tvr|charades>

python llm-verification/materialize_candidate_frames.py \
  --dataset "$DATASET" \
  --input outputs/upstream/pseudo_gt_candidates."$DATASET".jsonl \
  --video-root datasets/"$DATASET"/raw_videos \
  --workers 24
```

Expected raw-frame roots:

```text
activitynet: datasets/activitynet/raw_frames/<video_id>/*.jpg
tvr:         datasets/tvr/raw_frames/frames_hq/<show>_frames/<video_id>/*.jpg
charades:    datasets/charades/raw_frames/<video_id>/*.jpg
```

## 3. Verify Candidates

Run Qwen3-VL verification:

```bash
DATASET=<activitynet|tvr|charades>

python llm-verification/verify_pseudo_gt_with_qwen.py \
  --dataset "$DATASET" \
  --input outputs/upstream/pseudo_gt_candidates."$DATASET".jsonl \
  --output outputs/runs/"$DATASET"/verification.jsonl \
  --resume
```

Useful options:

```text
--model 4b|8b|30b-a3b
--limit <n>
--start-index <n>
--skip-missing-images
--plot
```

## 4. Export Dense Captions

Append accepted candidates to the validation caption file:

```bash
DATASET=<activitynet|tvr|charades>

python llm-verification/build_dense_caption.py \
  --dataset "$DATASET" \
  --split val \
  --caption-file datasets/"$DATASET"/TextData/"$DATASET"val.caption.txt \
  --verification outputs/runs/"$DATASET"/verification.jsonl \
  --output outputs/runs/"$DATASET"/"$DATASET"denseval.caption.txt \
  --overwrite
```

The accepted rows are also stored in:

```text
outputs/runs/<dataset>/extra_gt_all.jsonl
```
