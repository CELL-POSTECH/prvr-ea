# Dataset Setup

Set `PRVR_DATA_ROOT` to this directory (the default is `./datasets`).  All
paths below are relative to `$PRVR_DATA_ROOT`.

```bash
export PRVR_DATA_ROOT=/path/to/datasets
```

## Required layout

Feature-based PRVR experiments require the following files. `feature.bin`,
`id.txt`, `shape.txt`, and `video2frames.txt` must stay together in the shown
feature directory.

```text
datasets/
├── activitynet/
│   ├── FeatureData/
│   │   ├── i3d/{feature.bin,id.txt,shape.txt,video2frames.txt}
│   │   └── new_clip_vit_32_activitynet_vid_features.hdf5
│   └── TextData/
│       ├── activitynet{train,val,test}.caption.txt
│       ├── activitynet_{train,val}.jsonl
│       ├── roberta_activitynet_query_feat.hdf5
│       └── clip_ViT_B_32_activitynet_query_feat.hdf5
├── tvr/
│   ├── FeatureData/
│   │   ├── i3d_resnet/{feature.bin,id.txt,shape.txt,video2frames.txt}
│   │   └── new_clip_vit_32_tvr_vid_features.hdf5
│   └── TextData/
│       ├── tvr{train,val,test}.caption.txt
│       ├── tvr_{train,val}_release.jsonl
│       ├── roberta_tvr_query_feat.hdf5
│       └── clip_ViT_B_32_tvr_query_feat.hdf5
├── charades/
│   ├── FeatureData/
│   │   ├── i3d_rgb_lgi/{feature.bin,id.txt,shape.txt,video2frames.txt}
│   │   └── new_clip_vit_32_charades_vid_features.hdf5
│   └── TextData/
│       ├── charades{train,val,test}.caption.txt
│       ├── charades_{train,val}.jsonl
│       ├── roberta_charades_query_feat.hdf5
│       └── clip_ViT_B_32_charades_query_feat.hdf5
└── msrvtt/
    ├── FeatureData/
    │   ├── resnext101-resnet152/{feature.bin,id.txt,shape.txt,video2frames.txt}
    │   └── new_clip_vit_32_msrvtt_vid_features.hdf5
    ├── TextData/
    │   ├── msrvtt{train,val,test}.caption.txt
    │   ├── msrvtt_bert.pth.tar
    │   └── clip_ViT_B_32_msrvtt_query_feat.hdf5
    └── MSRVTT_data.videos.jsonl
```

Do not rename the feature directories used by the commands. In particular,
MSR-VTT must provide `FeatureData/resnext101-resnet152/` at the shown path.

## Download sources

| Asset | Source | Use |
| --- | --- | --- |
| PRVR feature bundle | [Google Drive](https://drive.google.com/drive/folders/11dRUeXmsWU25VMVmeuHc9nffzmZhPJEj) | Caption splits, ResNet/I3D features, RoBERTa features, and prepared CLIP-B/32 HDF5 features. See the [MS-SL data note](https://github.com/HuiGuanLab/ms-sl#required-data). |
| Candidate-mining annotations | [candidate_mining_annotations.zip](candidate_mining_annotations.zip) | Query-level annotation JSONL files used by candidate mining and MSR-VTT video metadata. |
| ActivityNet videos | [ActivityNet download page](http://activity-net.org/download.html) | Raw-frame extraction for CLIP4Clip; access may require registration. |
| TVR / TVQA videos | [TVQA download page](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html) | Raw-frame extraction for CLIP4Clip; follow TVQA access terms. |
| Charades videos | [Charades_v1.zip](https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1.zip) | Raw-frame extraction for CLIP4Clip. |
| MSR-VTT videos | [MSRVTT.zip](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip) | Raw-frame extraction for CLIP4Clip. |
| MSR-VTT ResNet features | [Hybrid Space MSRVTT10K](https://github.com/danieljf24/hybrid_space#dual-encoding-on-msrvtt10k) | `FeatureData/resnext101-resnet152/{feature.bin,id.txt,shape.txt,video2frames.txt}`. |
| MSR-VTT metadata | [msrvtt_data.zip](https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip) | CLIP4Clip metadata. |
| Verified dense multi-GT labels | [multigt_labels.zip](multigt_labels.zip) | Custom PRVR-candidate + Qwen3-VL verification output for dense evaluation. |

## Candidate Mining Annotations

Expand the candidate-mining annotation archive directly into `$PRVR_DATA_ROOT`:

```bash
unzip -q datasets/candidate_mining_annotations.zip -d "$PRVR_DATA_ROOT"
```

This provides:

```text
activitynet/TextData/activitynet_{train,val}.jsonl
tvr/TextData/tvr_{train,val}_release.jsonl
charades/TextData/charades_{train,val}.jsonl
msrvtt/MSRVTT_data.videos.jsonl
```

## Dense multi-GT evaluation

Dense evaluation adds verified multiple positive videos per query. Download the
archive above and expand it directly into `$PRVR_DATA_ROOT`:

```bash
unzip -q datasets/multigt_labels.zip -d "$PRVR_DATA_ROOT"
```

The resulting files must be at these paths; the evaluation scripts discover
them automatically:

```text
activitynet/activitynetdenseval.caption.txt
activitynet/activitynetdenseval.gt.jsonl
charades/charadesdenseval.caption.txt
charades/charadesdenseval.gt.jsonl
msrvtt/msrvttdenseval.caption.txt
msrvtt/msrvttdenseval.gt.jsonl
tvr/tvrdenseval_v.caption.txt
tvr/tvrdenseval_v.gt.jsonl
```

## Optional files

```text
# Required only by MSC-PRVR on TVR
annotations/tvr_train_release.jsonl
annotations/tvr_val_release.jsonl

# Required only by BOA vocabulary preprocessing
glove.840B.300d.txt
```

## Raw videos and frames

Feature-based PRVR train/eval does not use raw videos or raw frames.

CLIP4Clip PRVR evaluation reads JPEG frames from these paths:

```text
activitynet/raw_frames/<video_id>/*.jpg
charades/raw_frames/<video_id>/*.jpg
msrvtt/raw_frames/<video_id>/*.jpg
tvr/raw_frames/frames_hq/<show>_frames/<video_id>/*.jpg
```

TVR frames are distributed by TVQA. ActivityNet, Charades, and MSR-VTT are
distributed as raw videos; prepare the `raw_frames` directories before running
CLIP4Clip PRVR evaluation. For ActivityNet and Charades, sample raw videos at 3
fps and resize the shorter side to 224 pixels:

```text
activitynet: 1.5 fps
charades:    1.5 fps
```

LLM verification candidate mining uses CLIP video features for retrieval. It
also stores frame paths for the 8 GT frames and 8 pseudo frames used by the
verification step. These verification inputs are kept separate from CLIP4Clip
`raw_frames`.

Candidate mining writes verification-frame paths under:

```text
activitynet/verification_frames/<video_id>/*.jpg
tvr/verification_frames/<video_id>/*.jpg
charades/verification_frames/<video_id>/*.jpg
msrvtt/verification_frames/<video_id>/*.jpg
```

TVR verification frames are copied from the distributed raw frames:

```text
tvr/raw_frames/frames_hq/<show>_frames/<video_id>/*.jpg
```

ActivityNet, Charades, and MSR-VTT verification frames are materialized from
raw videos:

```text
activitynet/raw_videos/<video_file>
charades/raw_videos/<video_file>
msrvtt/raw_videos/<video_file>
```
