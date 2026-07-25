# Dataset Setup

Place all datasets under:

```text
$PRVR_DATA_ROOT
```

The recall scripts assume the directory names below exactly.

## Feature Data

Download the PRVR feature bundle from:

- https://drive.google.com/drive/folders/11dRUeXmsWU25VMVmeuHc9nffzmZhPJEj
- Source note: https://github.com/HuiGuanLab/ms-sl#required-data

Expected layout:

```text
datasets/
  activitynet/
    FeatureData/
      i3d/{feature.bin,id.txt,shape.txt,video2frames.txt}
      new_clip_vit_32_activitynet_vid_features.hdf5
    TextData/
      activitynettrain.caption.txt
      activitynetval.caption.txt
      activitynettest.caption.txt
      roberta_activitynet_query_feat.hdf5
      clip_ViT_B_32_activitynet_query_feat.hdf5

  tvr/
    FeatureData/
      i3d_resnet/{feature.bin,id.txt,shape.txt,video2frames.txt}
      new_clip_vit_32_tvr_vid_features.hdf5
    TextData/
      tvrtrain.caption.txt
      tvrval.caption.txt
      tvrtest.caption.txt
      roberta_tvr_query_feat.hdf5
      clip_ViT_B_32_tvr_query_feat.hdf5

  charades/
    FeatureData/
      i3d_rgb_lgi/{feature.bin,id.txt,shape.txt,video2frames.txt}
      new_clip_vit_32_charades_vid_features.hdf5
    TextData/
      charadestrain.caption.txt
      charadesval.caption.txt
      charadestest.caption.txt
      roberta_charades_query_feat.hdf5
      clip_ViT_B_32_charades_query_feat.hdf5

  msrvtt/
    FeatureData/
      resnext101-resnet152/{feature.bin,id.txt,shape.txt,video2frames.txt}
      new_clip_vit_32_msrvtt_vid_features.hdf5
    TextData/
      msrvtttrain.caption.txt
      msrvttval.caption.txt
      msrvtttest.caption.txt
      msrvtt_bert.pth.tar
      clip_ViT_B_32_msrvtt_query_feat.hdf5
```

`msrvtt/FeatureData/resnext101-resnet152` may be a symlink to the original long feature directory name.

## Dense Multi-GT

For dense multi-positive evaluation, add:

```text
datasets/
  activitynet/TextData/activitynetdenseval.gt.jsonl
  charades/TextData/charadesdenseval.gt.jsonl
  msrvtt/TextData/msrvttdenseval.gt.jsonl
  tvr/tvrdenseval_v.caption.txt
  tvr/tvrdenseval_v.gt.jsonl
```

TVR intentionally uses the `tvrdenseval_v.*` split.

## Extra Annotations

MSC-PRVR uses TVR release annotations:

```text
datasets/
  annotations/
    tvr_train_release.jsonl
    tvr_val_release.jsonl
```

BOA vocabulary setup uses:

```text
datasets/glove.840B.300d.txt
```

## Raw Frames

Feature-based PRVR eval does not require raw frames. CLIP4Clip zero-shot eval does.

Expected raw-frame layout:

```text
datasets/
  activitynet/raw_frames/<video_id>/*.jpg
  charades/raw_frames/<video_id>/*.jpg
  msrvtt/raw_frames/<video_id>/*.jpg
  tvr/raw_frames/frames_hq/<show>_frames/<video_id>/*.jpg
```

Raw video sources:

- Charades: https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip
- TVQA/TVR videos: https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html
- ActivityNet: http://activity-net.org/download.html
- MSR-VTT videos: https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
- MSR-VTT split/captions: https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip

TVQA and ActivityNet require dataset access forms from their providers.
