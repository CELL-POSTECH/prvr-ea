# PRVR EA

Reproducible training and evaluation for PRVR/T2VR retrieval.

```text
<repo-root>/
├── all_prvr/       # model code
├── datasets/       # dataset structure and setup guide
├── experiments/    # launchers and experiment utilities
├── llm-verification/# dense multi-GT label-generation pipeline
├── env.sh           # conda environment installer
└── env_faiss_gpu.sh # PRVR + ANN FAISS-GPU environment installer
```

Dataset files and download sources are described in
[datasets/README.md](datasets/README.md).

## Dense Multi-GT Label Generation

[`llm-verification/`](llm-verification/README.md) is the custom pipeline used
to mine PRVR candidates, verify them with Qwen3-VL, and export dense
multi-positive labels. The released labels are packaged for direct extraction
into the evaluator dataset layout:

```bash
unzip -q datasets/multigt_labels.zip -d datasets/
```

See [llm-verification/README.md](llm-verification/README.md) for the Qwen3-VL
environment and label-generation commands.

## Setup

Install the conda environment from the repository root:

```bash
bash env.sh
```

Set the project and data roots before running experiments:

```bash
export PRVR_PROJECT_ROOT="$PWD"
export PRVR_DATA_ROOT="$PRVR_PROJECT_ROOT/datasets"
export PRVR_PYTHON="python"
```

Datasets: `act`, `tvr`, `cha`, `msrvtt`, `all`.
Models: `AMDNet`, `GMMFormer`, `GMMFormer-v2`, `HLFormer`, `Holmes`,
`DreamPRVR`, `BOA`, `MSC-PRVR`, `DL-DKD`, `MS-SL`, `BGM-Net`, and
`CLIP4Clip` (CLIP only). MSC-PRVR is CLIP-only.

## Training

```bash
# ResNet/I3D features
bash experiments/scripts/train_resnet.sh all 0 all

# CLIP-B/32 features
bash experiments/scripts/train_clip.sh all 0 all

# One model / dataset condition
bash experiments/scripts/train_clip.sh act 0 GMMFormer-v2
```

Checkpoints are written below `all_prvr/<model>/results/<feature>/<dataset>/`.

## Eval

Evaluate standard single-GT recall from trained checkpoints. Results contain
`Method,R@1,R@5,R@10`.

```bash
bash experiments/scripts/eval_resnet.sh all 0 all
bash experiments/scripts/eval_clip.sh all 0 all
```

Outputs:

```text
experiments/recall_results/recall_{resnet,clip}_{tvr,act,cha,msrvtt}.csv
```

## Re-evaluate: Dense Multi-GT Recall

Re-evaluate the checkpoints using verified multi-positive ground truth.

```bash
bash experiments/scripts/eval_multigt_resnet.sh all 0 all
bash experiments/scripts/eval_multigt_clip.sh all 0 all
```

Outputs:

```text
experiments/recall_results/multiGT/recall_{resnet,clip}_<dataset>_multiGT.csv
```

## Dual-Branch Ablation

Evaluate CLIP checkpoints on ActivityNet and TVR under Base, clip-only,
frame-only, branch mean pooling, and weighted branch mean pooling.

```bash
bash experiments/scripts/eval_branch_ablation_clip.sh all 0 all
```

Outputs:

```text
experiments/branch_ablation/clip/{act,tvr}.csv
experiments/branch_ablation/clip/{act,tvr}_long.csv
```

## PRVR Latency

Measure query encoding and retrieval latency for CLIP4Clip and PRVR models on
an automatically generated synthetic CLIP corpus.

```bash
# All methods, 100K videos
bash experiments/scripts/latency.sh --gpu 0 100000
```

Outputs:

```text
experiments/latency_results/latency_detail.csv
experiments/latency_results/qps.csv
```

## PRVR + ANN Latency

Measure origin, IVF CPU, IVF GPU, and HNSW retrieval latency for static
dual-branch PRVR models. `*-x2` doubles only `nprobe` (IVF) or `efSearch`
(HNSW). This experiment requires the separate FAISS-GPU environment; create
and activate it before running the benchmark.

```bash
bash env_faiss_gpu.sh
conda activate prvr_faiss_gpu
export PRVR_PYTHON="python"
```

```bash
# ActivityNet, all supported models and seven conditions
for condition in origin ivf ivf-x2 ivf-gpu ivf-gpu-x2 hnsw hnsw-x2; do
  bash experiments/scripts/benchmark_static_dual_ann.sh "$condition" 0 all 0
done

# TVR
for condition in origin ivf ivf-x2 ivf-gpu ivf-gpu-x2 hnsw hnsw-x2; do
  PRVR_STATIC_ANN_DATASET=tvr \
    bash experiments/scripts/benchmark_static_dual_ann.sh "$condition" 0 all 0
done
```

Outputs are written to `experiments/ann_benchmark/<method>/<dataset>/` and
`experiments/ann_benchmark/static_dual_ann_<dataset>_<condition>.csv`.

## Recall with a Chunked Dataset

Run CLIP4Clip TVR zero-shot recall by splitting each original ordered raw-frame
sequence into chunks before any max-frame sampling. The chunk scores are
max-reduced to one score per parent video.

```bash
# One chunk size: 10, 20, or 30 sampled frames
bash experiments/scripts/eval_clip4clip_chunked_tvr.sh 0 10

# All chunk sizes
bash experiments/scripts/eval_clip4clip_chunked_tvr.sh 0 all
```

Outputs:

```text
all_prvr/CLIP4Clip/results/rawframes/tvr/zeroshot_f128_chunk{10,20,30}/
experiments/recall_results/recall_clip4clip_tvr_chunked.csv
```

If you encounter any issues, refer to the original model repositories for
implementation-specific details.
