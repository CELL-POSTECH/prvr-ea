"""Synthetic 512-D datasets for the GMMFormer-v2 speed-only evaluator.

This adapter preserves the model-facing tensors produced by the ordinary CLIP
loader while avoiding a 100k-video ``video2frames`` Python dictionary.  Each
HDF5 key is one video containing [frames, 512] CLIP-like features.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Datasets.data_provider import (
    average_to_fixed_length, collate_frame_val, collate_text_val,
    l2_normalize_np_array, uniform_feature_sampling,
)


class SyntheticVideoDataset(Dataset):
    def __init__(self, visual_path: Path, video_ids_path: Path, cfg):
        self.visual_path = str(visual_path)
        self.video_ids = [line.strip() for line in video_ids_path.open(encoding='utf-8') if line.strip()]
        if not self.video_ids:
            raise ValueError(f'empty synthetic context id file: {video_ids_path}')
        self.map_size = int(cfg['map_size'])
        self.max_ctx_len = int(cfg['max_ctx_l'])
        self._file = None
        self._pid = None
        with h5py.File(self.visual_path, 'r') as handle:
            first = handle[self.video_ids[0]]
            if first.shape[-1] != 512:
                raise ValueError(f'expected synthetic visual dim 512, got {first.shape}')

    def _handle(self):
        import os
        if self._file is None or self._pid != os.getpid():
            self._file = h5py.File(self.visual_path, 'r')
            self._pid = os.getpid()
        return self._file

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        frame_vecs = self._handle()[video_id][...]
        if frame_vecs.ndim != 2 or frame_vecs.shape[-1] != 512:
            raise ValueError(f'invalid synthetic visual feature for {video_id}: {frame_vecs.shape}')
        clip = average_to_fixed_length(frame_vecs, self.map_size)
        clip = torch.from_numpy(l2_normalize_np_array(clip)).unsqueeze(0)
        frame = uniform_feature_sampling(frame_vecs, self.max_ctx_len)
        frame = torch.from_numpy(l2_normalize_np_array(frame))
        return clip, frame, index, video_id

    def __len__(self):
        return len(self.video_ids)


class SyntheticTextDataset(Dataset):
    def __init__(self, caption_path: Path, text_path: Path, cfg):
        self.caption_ids = []
        with caption_path.open(encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.caption_ids.append(line.split(' ', 1)[0])
        if not self.caption_ids:
            raise ValueError(f'empty synthetic caption file: {caption_path}')
        self.text_path = str(text_path)
        self.max_desc_len = int(cfg['max_desc_l'])
        self._file = None
        self._pid = None

    def _handle(self):
        import os
        if self._file is None or self._pid != os.getpid():
            self._file = h5py.File(self.text_path, 'r')
            self._pid = os.getpid()
        return self._file

    def __getitem__(self, index):
        caption_id = self.caption_ids[index]
        feature = np.atleast_2d(self._handle()[caption_id][...])
        feature = torch.from_numpy(l2_normalize_np_array(feature))[:self.max_desc_len]
        return feature, index, caption_id

    def __len__(self):
        return len(self.caption_ids)


def get_synthetic_speed_datasets(cfg, args):
    root = Path(args.synthetic_data_root or cfg['data_root'])
    collection = cfg['collection']
    feature_root = root / collection / 'FeatureData'
    text_root = root / collection / 'TextData'
    visual_path = feature_root / f'new_clip_vit_32_{collection}_vid_features.hdf5'
    caption_path = text_root / f'{collection}test.caption.txt'
    text_path = text_root / f'clip_ViT_B_32_{collection}_query_feat.hdf5'
    context_ids = Path(args.synthetic_context_ids) if args.synthetic_context_ids else \
        text_root / 'synthetic_test_video_ids.txt'
    for path in (visual_path, caption_path, text_path, context_ids):
        if not path.exists():
            raise FileNotFoundError(f'synthetic speed input missing: {path}')
    video_dataset = SyntheticVideoDataset(visual_path, context_ids, cfg)
    text_dataset = SyntheticTextDataset(caption_path, text_path, cfg)
    workers = int(cfg['num_workers'])
    return (
        DataLoader(video_dataset, batch_size=int(cfg['eval_context_bsz']), shuffle=False,
                   num_workers=workers, pin_memory=cfg['pin_memory'], collate_fn=collate_frame_val),
        DataLoader(text_dataset, batch_size=int(cfg['eval_query_bsz']), shuffle=False,
                   num_workers=workers, pin_memory=cfg['pin_memory'], collate_fn=collate_text_val),
    )
