# CLIP4Clip with PRVR raw frames

The original CLIP4Clip model and its retrieval/loss logic are unchanged.  The
`raw_*` datatypes add a data adapter that reads each video's already-decoded
images instead of decoding an MP4 at runtime.  Frame preprocessing, ordering,
and `--slice_framepos` behavior follow CLIP4Clip's original raw-video loader.

Supported datasets and frame roots:

| Dataset | datatype | frame root |
| --- | --- | --- |
| MSRVTT | `raw_msrvtt` | `$PRVR_DATA_ROOT/msrvtt/raw_frames` |
| ActivityNet | `raw_activitynet` | `$PRVR_DATA_ROOT/activitynet/raw_frames` |
| TVR | `raw_tvr` | `$PRVR_DATA_ROOT/tvr/raw_frames/frames_hq` |
| Charades | `raw_charades` | `$PRVR_DATA_ROOT/charades/raw_frames` |

Each frame root must contain one directory per video. TVR has one additional
show directory level: `frames_hq/{show}_frames/{video_id}/*.jpg`.

## Commands

Run a zero-shot CLIP4Clip evaluation using uniformly sampled 128 frames:

```bash
cd $PRVR_PROJECT_ROOT/all_prvr/CLIP4Clip
bash scripts/run_prvr_rawframes.sh zeroshot msrvtt 0
bash scripts/run_prvr_rawframes.sh zeroshot activitynet 0
bash scripts/run_prvr_rawframes.sh zeroshot tvr 0
bash scripts/run_prvr_rawframes.sh zeroshot charades 0
```

Train and evaluate a saved checkpoint:

```bash
bash scripts/run_prvr_rawframes.sh train msrvtt 0
bash scripts/run_prvr_rawframes.sh eval msrvtt 0 \
  results/rawframes/msrvtt/train_f128/pytorch_model.bin.4
```

The helper defaults to `MAX_FRAMES=128`, `BATCH_SIZE=8`, and
`BATCH_SIZE_VAL=2` for a conservative single-GPU run. Override them explicitly,
for example `MAX_FRAMES=64 BATCH_SIZE=16 BATCH_SIZE_VAL=4 ...`.

For a loader/model smoke test only, add `MAX_EVAL_SAMPLES=8`; the default `0`
always uses the complete split.

Results and checkpoints are stored below `results/rawframes/<dataset>/`.
