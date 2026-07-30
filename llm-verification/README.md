# Dense Multi-GT Labeling Pipeline

This directory contains the custom pipeline used to mine TREC-style candidate
moments from PRVR models and verify them with an LLM.

The released labels are packaged at `../datasets/multigt_labels.zip`. To use
them with the PRVR evaluation scripts from the repository root:

```bash
unzip -q datasets/multigt_labels.zip -d datasets/
```

The code is organized around three steps:

1. Mine candidate moments from trained PRVR checkpoints.
2. Materialize or check the candidate frames.
3. Run LLM verification and export accepted dense labels.

Generated candidates, verification outputs, plots, datasets, and checkpoints are
not committed.

## Setup

Create the LLM verification environment:

```bash
./setup_qwen3vl.sh
```

The main Python dependencies are also listed in:

```text
requirements-qwen3vl.txt
environment-qwen3vl.yml
```

## Expected Layout

Use local paths that match your machine. The examples below use relative
directories only.

```text
datasets/
  activitynet/
  tvr/
outputs/
  upstream/
  runs/
prvr_models/
  <dataset>/<run>/pytorch_model.bin.best
```

Candidate files are expected at:

```text
outputs/upstream/pseudo_gt_candidates.<dataset>.jsonl
```

The candidate-mining step requires the PRVR model code, dataset files, raw
frames, and trained checkpoints. Place the trained PRVR checkpoints under a
local model directory such as `prvr_models/`.

## Step 1: Mine Candidates

Run the PRVR candidate miner for each dataset. The miner should write one JSONL
file per dataset:

```text
outputs/upstream/pseudo_gt_candidates.tvr.jsonl
outputs/upstream/pseudo_gt_candidates.activitynet.jsonl
```

Each row should contain the query, the original ground-truth frame paths, the
candidate video id, and the candidate frame paths. The verification scripts read
these files directly.

## Step 2: Prepare Frames

TVR candidates normally point to extracted frame files:

```text
datasets/tvr/raw_frames/frames_hq/<show>_frames/<video_id>/<frame>.jpg
```

ActivityNet candidates may point to timestamped frame paths. Materialize them
from raw videos before verification:

```bash
python materialize_candidate_frames.py \
  --dataset activitynet \
  --video-root datasets/activitynet/raw_videos \
  --workers 24
```

If candidate paths need to be rewritten for another machine:

```bash
python materialize_candidate_frames.py \
  --dataset activitynet \
  --video-root datasets/activitynet/raw_videos \
  --path-prefix-from datasets/activitynet/raw_frames \
  --path-prefix-to /path/on/server/activitynet/raw_frames \
  --output-jsonl outputs/upstream/pseudo_gt_candidates.activitynet.server.jsonl \
  --workers 8
```

Check candidate frame paths before a long verification run:

```bash
python sanity_check_candidate_frames.py \
  --dataset activitynet \
  --caption-file datasets/activitynet/activitynetval.caption.txt \
  --workers 32 \
  --require-video-set-match \
  --fail-on-issue
```

## Step 3: Verify Candidates

Run LLM verification:

```bash
conda run -n qwen3vl python verify_pseudo_gt_with_qwen.py \
  --dataset activitynet \
  --max-pixels 112896 \
  --resume
```

For TVR:

```bash
conda run -n qwen3vl python verify_pseudo_gt_with_qwen.py \
  --dataset tvr \
  --resume
```

Useful options:

```text
--input <path>          Override the default candidate JSONL.
--output <path>         Override the verification output path.
--model 4b|8b|30b-a3b   Select the Qwen3-VL model size.
--limit <n>             Run a small subset.
--start-index <n>       Start from an input line.
--skip-missing-images   Record missing images instead of stopping.
--plot                  Render verification galleries.
```

Default outputs:

```text
outputs/runs/<dataset>/verification.jsonl
outputs/runs/<dataset>/summary.json
outputs/runs/<dataset>/extra_gt_all.jsonl
```

## Export Dense Captions

After verification, accepted rows can be appended to a caption file:

```bash
python build_dense_caption.py \
  --dataset activitynet \
  --split val \
  --caption-file datasets/activitynet/activitynetval.caption.txt \
  --verification outputs/runs/activitynet/verification.jsonl \
  --output outputs/runs/activitynet/activitynetdenseval.caption.txt \
  --overwrite
```

## Plot Results

```bash
python plot_tvr_qwen_verification.py \
  --input outputs/runs/activitynet/verification.jsonl \
  --output-dir outputs/runs/activitynet/plots \
  --max-pixels 112896 \
  --backend pil \
  --overwrite
```

## Artifact Notes

- Do not commit datasets, model checkpoints, generated candidates, or run
  outputs.
- Use a neutral repository name for anonymous review.
- Do not include personal paths, usernames, hostnames, API keys, or access
  tokens in committed files.
- If a web page is used for the artifact, add `noindex` metadata.
- Do not use shortened tracking links for the artifact URL.
