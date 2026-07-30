# Dataset Setup

Set `PRVR_DATA_ROOT` to this directory (the default is `./datasets`).  All
paths below are relative to `$PRVR_DATA_ROOT`.

```bash
export PRVR_DATA_ROOT=/path/to/prvr-experiments/datasets
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
│       ├── roberta_activitynet_query_feat.hdf5
│       └── clip_ViT_B_32_activitynet_query_feat.hdf5
├── tvr/
│   ├── FeatureData/
│   │   ├── i3d_resnet/{feature.bin,id.txt,shape.txt,video2frames.txt}
│   │   └── new_clip_vit_32_tvr_vid_features.hdf5
│   └── TextData/
│       ├── tvr{train,val,test}.caption.txt
│       ├── roberta_tvr_query_feat.hdf5
│       └── clip_ViT_B_32_tvr_query_feat.hdf5
├── charades/
│   ├── FeatureData/
│   │   ├── i3d_rgb_lgi/{feature.bin,id.txt,shape.txt,video2frames.txt}
│   │   └── new_clip_vit_32_charades_vid_features.hdf5
│   └── TextData/
│       ├── charades{train,val,test}.caption.txt
│       ├── roberta_charades_query_feat.hdf5
│       └── clip_ViT_B_32_charades_query_feat.hdf5
└── msrvtt/
    ├── FeatureData/
    │   ├── resnext101-resnet152/{feature.bin,id.txt,shape.txt,video2frames.txt}
    │   └── new_clip_vit_32_msrvtt_vid_features.hdf5
    └── TextData/
        ├── msrvtt{train,val,test}.caption.txt
        ├── msrvtt_bert.pth.tar
        └── clip_ViT_B_32_msrvtt_query_feat.hdf5
```

`resnext101-resnet152` may be a symlink to an equivalent MSR-VTT ResNet
feature directory. Do not rename the directory used by the commands.

## Download sources

| Asset | Source | Use |
| --- | --- | --- |
| PRVR feature bundle | [Google Drive](https://drive.google.com/drive/folders/11dRUeXmsWU25VMVmeuHc9nffzmZhPJEj) | Caption splits, ResNet/I3D features, RoBERTa features, and prepared CLIP-B/32 HDF5 features. See the [MS-SL data note](https://github.com/HuiGuanLab/ms-sl#required-data). |
| ActivityNet videos | [ActivityNet download page](http://activity-net.org/download.html) | Raw-frame extraction for CLIP4Clip; access may require registration. |
| TVR / TVQA videos | [TVQA download page](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html) | Raw-frame extraction for CLIP4Clip; follow TVQA access terms. |
| Charades videos | [Charades_v1.zip](https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1.zip) | Raw-frame extraction for CLIP4Clip. |
| MSR-VTT videos | [MSRVTT.zip](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip) | Raw-frame extraction for CLIP4Clip. |
| MSR-VTT metadata | [msrvtt_data.zip](https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip) | CLIP4Clip metadata. |
| Verified dense multi-GT labels | [multigt_labels.zip](multigt_labels.zip) | Custom PRVR-candidate + Qwen3-VL verification output for dense evaluation. |

## Dense multi-GT evaluation

Dense evaluation adds verified multiple positive videos per query. Download the
archive above and expand it directly into `datasets/`:

```bash
unzip -q datasets/multigt_labels.zip -d datasets/
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

## Raw frames for CLIP4Clip

Feature-based PRVR models do not use raw frames. CLIP4Clip reads JPEG frames
from these paths:

```text
activitynet/raw_frames/<video_id>/*.jpg
charades/raw_frames/<video_id>/*.jpg
msrvtt/raw_frames/<video_id>/*.jpg
tvr/raw_frames/frames_hq/<show>_frames/<video_id>/*.jpg
```

The frame loader samples up to the configured `--max_frames` from each video.
