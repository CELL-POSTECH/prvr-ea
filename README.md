# PRVR Reproduction

Reproducible training and evaluation for PRVR/T2VR retrieval with ResNet/I3D and CLIP-B/32 features.

```text
<repo-root>/
├── all_prvr/       # model code
├── datasets/       # dataset structure and setup guide
└── experiments/    # launchers and experiment utilities
```

Dataset layout is described in [datasets/README.md](datasets/README.md).

## Setup

Run from the repository root. Override these paths when needed.

```bash
export PRVR_PROJECT_ROOT="$PWD"
export PRVR_DATA_ROOT="$PRVR_PROJECT_ROOT/datasets"
export PRVR_PYTHON="python"
```

## Models

ResNet/I3D: AMDNet, GMMFormer, GMMFormer-v2, HLFormer, Holmes, DreamPRVR, BOA, DL-DKD, MS-SL, BGM-Net.

CLIP: CLIP4Clip, AMDNet, GMMFormer, GMMFormer-v2, HLFormer, Holmes, DreamPRVR, BOA, MSC-PRVR, DL-DKD, MS-SL, BGM-Net.

- CLIP4Clip is zero-shot raw-frame evaluation only.
- MSC-PRVR is CLIP-only.
- DL-DKD retains its original ResNet/I3D student and CLIP-B/32 teacher under both feature tables.

All train/eval launchers use:

```text
<dataset|all> <gpu> [model|all]
```

Datasets: `act`, `tvr`, `cha`, `msrvtt`, `all`.

## Train

```bash
# All compatible models and datasets.
bash experiments/scripts/train_resnet.sh all 0 all
bash experiments/scripts/train_clip.sh all 1 all

# One condition.
bash experiments/scripts/train_resnet.sh act 0 GMMFormer-v2
bash experiments/scripts/train_clip.sh msrvtt 1 MSC-PRVR
```

Checkpoints are written below `all_prvr/<model>/results/<feature>/<dataset>/`.

## Single-GT Recall

```bash
bash experiments/scripts/eval_resnet.sh all 0 all
bash experiments/scripts/eval_clip.sh all 1 all
```

The launchers evaluate available checkpoints and export one CSV per dataset:

```text
experiments/recall_results/recall_{resnet,clip}_{tvr,act,cha,msrvtt}.csv
```

Each CSV has `Method,R@1,R@5,R@10`.

## Dense Multi-GT Recall

```bash
bash experiments/scripts/eval_multigt_resnet.sh all 2 all
bash experiments/scripts/eval_multigt_clip.sh all 3 all
```

Results are written to:

```text
experiments/recall_results/multiGT/recall_{resnet,clip}_<dataset>_multiGT.csv
```

TVR dense evaluation uses `datasets/tvr/tvrdenseval_v.caption.txt` and `datasets/tvr/tvrdenseval_v.gt.jsonl`.

## CLIP Dual-Branch Ablation

The ablation evaluates CLIP checkpoints on ActivityNet and TVR under Base, clip-only, frame-only, branch mean pooling, and weighted branch mean pooling.

```bash
bash experiments/scripts/eval_branch_ablation_clip.sh all 0 all
bash experiments/scripts/eval_branch_ablation_clip.sh act 0 GMMFormer-v2
```

Outputs:

```text
experiments/branch_ablation/clip/{act,tvr}.csv
experiments/branch_ablation/clip/{act,tvr}_long.csv
```

## Synthetic Retrieval Latency

This benchmark measures CLIP4Clip and all PRVR methods over synthetic 512-D CLIP source features. Gallery/context encoding is offline preparation; reported E2E latency is query encoding plus retrieval over the prepared gallery.

```bash
# All methods, 100K videos. The corpus is created automatically.
bash experiments/scripts/latency.sh --gpu 0 100000

# 100K, 500K, 1M, and 5M videos.
bash experiments/scripts/latency.sh --gpu 0 --100k --500k --1m --5m

# One method.
bash experiments/scripts/latency.sh --gpu 0 --method GMMFormerV2 100000
```

Outputs:

```text
experiments/latency_results/latency_detail.csv
experiments/latency_results/qps.csv
```

## GMMFormer-v2 ANN Latency

The benchmark measures retrieval only: clip search, frame search, deduplication, cross-branch rerank, fusion/top-10, and total time. Query/video encoding is outside the timing region.

```bash
# Original GMMFormer-v2 evaluation context pipeline; no context bank.
bash experiments/scripts/benchmark_gmmformer_v2_ann.sh origin 0

# IVF/HNSW use the pre-encoded context bank.
bash experiments/scripts/build_gmmformer_v2_context_bank.sh act 0
bash experiments/scripts/benchmark_gmmformer_v2_ann.sh ivf 0
bash experiments/scripts/benchmark_gmmformer_v2_ann.sh ivf-gpu 0
bash experiments/scripts/benchmark_gmmformer_v2_ann.sh hnsw 0
```

ANN outputs are stored under `experiments/ann_benchmark/GMMFormer_v2/act/`.

## Generated Files

Checkpoints, feature files, raw frames, logs, indices, and experiment CSVs are generated artifacts and should not be committed. The repository `.gitignore` excludes them.
