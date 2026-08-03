"""Shared, non-model compatibility helpers for the local PRVR experiments.

This module deliberately only selects files and adapts their storage format.  It
does not alter any model, loss, sampler, or retrieval calculation.
"""
import os


PROJECT_ROOT = os.environ.get("PRVR_PROJECT_ROOT",
                              os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_ROOT = os.environ.get("PRVR_DATA_ROOT", os.path.join(PROJECT_ROOT, "datasets"))
MSRVTT_RESNET_FEATURE = "resnext101-resnet152"


def legacy_feature_dim(collection, feature_name):
    """Read the authoritative BigFile embedding size from ``shape.txt``."""
    shape_file = os.path.join(DATA_ROOT, collection, "FeatureData", feature_name,
                              "shape.txt")
    with open(shape_file, "r", encoding="utf-8") as file:
        return int(file.readline().split()[1])


def text_feature_dim(collection, feature_mode):
    """Read the supplied query embedding size without loading all features."""
    if collection == "msrvtt" and feature_mode == "resnet":
        filename = "msrvtt10k_cap_feat.hdf5"
    else:
        prefix = "clip_ViT_B_32" if feature_mode == "clip" else "roberta"
        filename = "%s_%s_query_feat.hdf5" % (prefix, collection)
    import h5py
    path = os.path.join(DATA_ROOT, collection, "TextData", filename)
    with h5py.File(path, "r") as file:
        return int(file[next(iter(file.keys()))].shape[-1])


def dataset_from_cli(name):
    """Return (collection, feature_mode) for `act` / `act_clip` style names."""
    feature_mode = "clip" if name.endswith("_clip") else "resnet"
    base = name[:-5] if feature_mode == "clip" else name
    aliases = {"act": "activitynet", "activitynet": "activitynet",
               "msrvtt": "msrvtt", "tvr": "tvr", "cha": "charades"}
    if base not in aliases:
        raise ValueError("Unknown PRVR dataset alias: %s" % name)
    return aliases[base], feature_mode


def frame_index_feature_for(collection, feature_mode):
    if feature_mode != "clip":
        return None
    clip_frame_maps = {
        "activitynet": "i3d",
        "tvr": "i3d_resnet",
        "charades": "i3d_rgb_lgi",
        "msrvtt": MSRVTT_RESNET_FEATURE,
    }
    return clip_frame_maps[collection]


def configure_cfg(cfg, cli_name, project_root):
    """Apply local paths and feature selection to config-dict based projects."""
    collection, feature_mode = dataset_from_cli(cli_name)
    cfg["root"] = project_root
    cfg["dataset_name"] = collection
    cfg["collection"] = collection
    cfg["data_root"] = DATA_ROOT
    cfg["feature_mode"] = feature_mode
    legacy_visual_features = {
        "activitynet": "i3d",
        "tvr": "i3d_resnet",
        "charades": "i3d_rgb_lgi",
        "msrvtt": MSRVTT_RESNET_FEATURE,
    }
    cfg["visual_feature"] = "clip" if feature_mode == "clip" else \
        legacy_visual_features[collection]
    # CLIP HDF5 stores per-video sequences, but the PRVR loaders still need the
    # original frame index map to preserve each dataset's temporal ordering.
    cfg["frame_index_feature"] = frame_index_feature_for(collection, feature_mode) \
        if feature_mode == "clip" else cfg["visual_feature"]
    if feature_mode == "clip":
        # Supplied CLIP ViT-B/32 video and text embeddings are 512-D.
        cfg["visual_feat_dim"] = 512
        cfg["q_feat_size"] = 512
        if cfg.get("model_name") == "boa":
            # BOA distinguishes raw text input from its 384-D semantic-bank
            # representation.  Its q_feat_size is the latter.
            cfg["text_feat_dim"] = 512
            cfg["q_feat_size"] = cfg["hidden_size"]
    elif cfg.get("model_name") in {"model_name", "AMDNet", "N_np"}:
        # AMDNet ships with CLIP inputs; its projection layers also support the
        # 1024-D ActivityNet I3D/Roberta pair when configured explicitly.
        cfg["visual_feat_dim"] = 1024
        cfg["q_feat_size"] = text_feature_dim(collection, feature_mode)
    if collection == "msrvtt" and feature_mode == "resnet":
        cfg["visual_feat_dim"] = legacy_feature_dim(collection, cfg["visual_feature"])
        cfg["q_feat_size"] = text_feature_dim(collection, feature_mode)
    elif feature_mode == "resnet":
        cfg["visual_feat_dim"] = legacy_feature_dim(collection, cfg["visual_feature"])
    if feature_mode == "resnet" and cfg.get("model_name") != "boa":
        cfg["q_feat_size"] = text_feature_dim(collection, feature_mode)
    if cfg.get("model_name") == "boa" and feature_mode == "resnet":
        cfg["text_feat_dim"] = text_feature_dim(collection, feature_mode)
        cfg["q_feat_size"] = cfg["hidden_size"]
    result_root = os.path.join(project_root, "results", feature_mode, collection,
                               cfg["model_name"])
    cfg["model_root"] = result_root
    cfg["ckpt_path"] = os.path.join(result_root, "ckpt")
    if "tb_dir" in cfg:
        cfg["tb_dir"] = result_root
    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    return cfg


class H5FrameFeature:
    """Expose per-video HDF5 embeddings through the BigFile `read_one` API.

    Frame identifiers in the legacy BigFile `video2frames.txt` are converted to
    ``(<video id>, <frame index>)``.  If the feature extractors produced a
    different number of temporal samples, the index is safely clamped; this is
    the same temporal resampling subsequently performed by the data loaders.
    Each worker opens its own HDF5 handle, making this safe with DataLoader.
    """
    def __init__(self, path):
        self.path = path
        self._pid = None
        self._file = None
        self._legacy_last_index = {}
        import h5py
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            if not keys:
                raise ValueError("Empty visual feature HDF5: %s" % path)
            first = f[keys[0]]
            self.ndims = int(first.shape[-1])

    def set_frame_index_map(self, video2frames):
        """Map legacy raw-frame indices onto each HDF5 video's sample count."""
        self._legacy_last_index = {
            video_id: max((int(frame.rsplit('_', 1)[1]) for frame in frames
                           if '_' in frame and frame.rsplit('_', 1)[1].isdigit()), default=0)
            for video_id, frames in video2frames.items()
        }
        return self

    def _handle(self):
        if self._file is None or self._pid != os.getpid():
            import h5py
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def read_one(self, frame_id):
        video_id, index = frame_id.rsplit("_", 1)
        values = self._handle()[video_id]
        index = int(index)
        legacy_last = self._legacy_last_index.get(video_id, 0)
        if legacy_last > 0:
            index = round(index / legacy_last * (len(values) - 1))
        return values[min(index, len(values) - 1)][...].tolist()


def visual_paths(root, collection, feature_mode, visual_feature):
    """Return (visual reader/path, video2frames path) for legacy loaders."""
    feature_root = os.path.join(root, collection, "FeatureData")
    frame_index_feature = frame_index_feature_for(collection, feature_mode) \
        if feature_mode == "clip" else visual_feature
    frame_map = os.path.join(feature_root, frame_index_feature, "video2frames.txt")
    if feature_mode == "clip":
        return H5FrameFeature(os.path.join(
            feature_root, "new_clip_vit_32_%s_vid_features.hdf5" % collection)), frame_map
    return None, frame_map


def text_feature_path(root, collection, feature_mode):
    if collection == "msrvtt" and feature_mode == "resnet":
        # `msrvtt_bert.pth.tar` is a gzip archive of this same per-caption
        # feature payload; the extracted HDF5 is directly consumable here.
        return os.path.join(root, collection, "TextData", "msrvtt10k_cap_feat.hdf5")
    prefix = "clip_ViT_B_32" if feature_mode == "clip" else "roberta"
    return os.path.join(root, collection, "TextData", "%s_%s_query_feat.hdf5" % (prefix, collection))
