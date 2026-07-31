from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import json
import os
from collections import OrderedDict

import numpy as np
from torch.utils.data import Dataset

from dataloaders.rawvideo_util import RawFrameExtractor


_PROJECT_ROOT = os.environ.get(
    "PRVR_PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
_DATA_ROOT = os.environ.get("PRVR_DATA_ROOT", os.path.join(_PROJECT_ROOT, "datasets"))

DEFAULT_DATA_ROOTS = {
    "msrvtt": os.path.join(_DATA_ROOT, "msrvtt"),
    "activitynet": os.path.join(_DATA_ROOT, "activitynet"),
    "tvr": os.path.join(_DATA_ROOT, "tvr"),
    "charades": os.path.join(_DATA_ROOT, "charades"),
}
DEFAULT_FRAME_ROOTS = {
    "msrvtt": os.path.join(_DATA_ROOT, "msrvtt", "raw_frames"),
    "activitynet": os.path.join(_DATA_ROOT, "activitynet", "raw_frames"),
    "tvr": os.path.join(_DATA_ROOT, "tvr", "raw_frames", "frames_hq"),
    "charades": os.path.join(_DATA_ROOT, "charades", "raw_frames"),
}


class PRVRRawFramesRetrievalDataset(Dataset):
    """CLIP4Clip retrieval dataset backed by PRVR caption files and decoded frames.

    The tensor layout, CLIP preprocessing, and frame-order handling are the same
    as CLIP4Clip's raw-video loaders.  Only video decoding is replaced by reads
    from an already-decoded per-video frame directory.
    """

    SPECIAL_TOKEN = {
        "CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
        "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]",
    }

    def __init__(self, collection, subset, tokenizer, data_path=None, frame_root=None,
                 caption_root=None, max_words=30, feature_framerate=1.0,
                 max_frames=100, image_resolution=224, frame_order=0,
                 slice_framepos=0, max_samples=0, multi_gt_eval=False,
                 multi_gt_caption_file=None, multi_gt_file=None, chunk_size=0):
        if collection not in DEFAULT_DATA_ROOTS:
            raise ValueError("unsupported raw-frame collection: {}".format(collection))
        if subset not in ("train", "val", "test"):
            raise ValueError("unsupported split: {}".format(subset))
        if frame_order not in (0, 1, 2) or slice_framepos not in (0, 1, 2):
            raise ValueError("frame_order and slice_framepos must be 0, 1, or 2")

        self.collection = collection
        self.subset = subset
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos
        self.chunk_size = int(chunk_size)
        self.prechunk_eval = self.chunk_size > 0 and subset in ("val", "test")
        if self.chunk_size < 0:
            raise ValueError("chunk_size must be non-negative")
        if self.prechunk_eval and self.chunk_size > self.max_frames:
            raise ValueError("chunk_size must not exceed max_frames")
        self.data_path = data_path or DEFAULT_DATA_ROOTS[collection]
        self.caption_root = caption_root or os.path.join(self.data_path, "TextData")
        self.frame_root = frame_root or DEFAULT_FRAME_ROOTS[collection]
        self.multi_gt_eval = multi_gt_eval and subset in ("val", "test")
        if self.multi_gt_eval and collection == "tvr":
            self.caption_path = multi_gt_caption_file or os.path.join(
                self.data_path, "tvrdenseval_v.caption.txt"
            )
            self.multi_gt_file = multi_gt_file or os.path.join(
                self.data_path, "tvrdenseval_v.gt.jsonl"
            )
        elif self.multi_gt_eval:
            self.caption_path = multi_gt_caption_file or os.path.join(
                self.caption_root, "{}denseval.caption.txt".format(collection)
            )
            self.multi_gt_file = multi_gt_file or os.path.join(
                self.caption_root, "{}denseval.gt.jsonl".format(collection)
            )
        else:
            self.caption_path = os.path.join(
                self.caption_root, "{}{}.caption.txt".format(collection, subset)
            )
            self.multi_gt_file = None
        # Some dense releases provide only the GT JSONL. In that case, keep
        # using the ordinary PRVR split captions as model inputs.
        if self.multi_gt_eval and not os.path.isfile(self.caption_path):
            self.caption_path = os.path.join(
                self.caption_root, "{}{}.caption.txt".format(collection, subset)
            )
        if not os.path.isfile(self.caption_path):
            raise FileNotFoundError("caption file not found: {}".format(self.caption_path))
        if not os.path.isdir(self.frame_root):
            raise FileNotFoundError("raw-frame root not found: {}".format(self.frame_root))
        if self.multi_gt_eval and not os.path.isfile(self.multi_gt_file):
            raise FileNotFoundError("multi-GT file not found: {}".format(self.multi_gt_file))

        rows = self._load_caption_rows(self.caption_path)
        if max_samples > 0 and not self.multi_gt_eval:
            rows = rows[:max_samples]
        if self.multi_gt_eval:
            gt_query_ids = self._load_multi_gt_query_ids(self.multi_gt_file)
            if max_samples > 0:
                gt_query_ids = gt_query_ids[:max_samples]
            row_by_query_id = OrderedDict((row["query_id"], row) for row in rows)
            candidate_video_rows = OrderedDict()
            for row in rows:
                candidate_video_rows.setdefault(row["video_id"], row)
            query_rows = [row_by_query_id[qid] for qid in gt_query_ids if qid in row_by_query_id]
            if not query_rows:
                raise ValueError("no dense multi-GT queries from {} were found in {}".format(
                    self.multi_gt_file, self.caption_path))
            # Keep every dense-caption candidate video.  A single representative
            # caption is retained for videos with no selected dense query so that
            # CLIP4Clip encodes the full candidate collection exactly once/video.
            self.video_to_rows = OrderedDict((video_id, []) for video_id in candidate_video_rows)
            for row in query_rows:
                self.video_to_rows.setdefault(row["video_id"], []).append(row)
            for video_id, row in candidate_video_rows.items():
                if not self.video_to_rows[video_id]:
                    self.video_to_rows[video_id].append(row)
            self.multi_gt_query_id_set = {row["query_id"] for row in query_rows}
        else:
            self.video_to_rows = OrderedDict()
            for row in rows:
                self.video_to_rows.setdefault(row["video_id"], []).append(row)

        if subset in ("val", "test"):
            self.query_rows = []
            self.cut_off_points = []
            for video_rows in self.video_to_rows.values():
                self.query_rows.extend(video_rows)
                self.cut_off_points.append(len(self.query_rows))
            if self.multi_gt_eval:
                self.eval_query_indices = [
                    idx for idx, row in enumerate(self.query_rows)
                    if row["query_id"] in self.multi_gt_query_id_set
                ]
                self.eval_query_ids = [self.query_rows[idx]["query_id"] for idx in self.eval_query_indices]
                self.eval_video_ids = list(self.video_to_rows)
            self.multi_sentence_per_video = True
        else:
            self.query_rows = rows
            self.cut_off_points = []
            self.multi_sentence_per_video = False

        self.sentence_num = len(self.query_rows)
        self.video_num = len(self.video_to_rows)
        self.rawFrameExtractor = RawFrameExtractor(framerate=feature_framerate, size=image_resolution)
        print("{} {} captions: {}, videos: {}, frame_root: {}".format(
            collection, subset, self.sentence_num, self.video_num, self.frame_root
        ))

    @staticmethod
    def _load_caption_rows(caption_path):
        rows = []
        with open(caption_path, "r", encoding="utf-8") as reader:
            for line_no, line in enumerate(reader, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    raise ValueError("invalid caption line {} in {}".format(line_no, caption_path))
                query_id, caption = parts
                rows.append({
                    "query_id": query_id,
                    "video_id": query_id.split("#", 1)[0],
                    "caption": caption,
                })
        return rows

    @staticmethod
    def _load_multi_gt_query_ids(gt_path):
        query_ids, seen = [], set()
        with open(gt_path, "r", encoding="utf-8") as reader:
            for line_no, line in enumerate(reader, start=1):
                line = line.strip()
                if not line:
                    continue
                query_id = json.loads(line).get("query_id")
                if not query_id:
                    raise ValueError("missing query_id at line {} in {}".format(line_no, gt_path))
                if query_id not in seen:
                    query_ids.append(query_id)
                    seen.add(query_id)
        return query_ids

    def __len__(self):
        return len(self.query_rows)

    def _get_text(self, caption):
        pairs_text = np.zeros((1, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((1, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((1, self.max_words), dtype=np.int64)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + self.tokenizer.tokenize(caption)
        words = words[:self.max_words - 1] + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        pairs_text[0], pairs_mask[0], pairs_segment[0] = input_ids, input_mask, segment_ids
        return pairs_text, pairs_mask, pairs_segment

    def _resolve_frame_dir(self, video_id):
        candidates = [os.path.join(self.frame_root, video_id)]
        if self.collection == "tvr":
            show = video_id.split("_", 1)[0]
            candidates.insert(0, os.path.join(self.frame_root, "{}_frames".format(show), video_id))
            candidates.extend(glob.glob(os.path.join(self.frame_root, "*_frames", video_id)))
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError("frame directory for {} not found under {}".format(video_id, self.frame_root))

    def _get_rawframes(self, video_id):
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        video = np.zeros((1, self.max_frames, 1, 3,
                          self.rawFrameExtractor.size, self.rawFrameExtractor.size), dtype=np.float32)
        frame_data = self.rawFrameExtractor.get_frame_data(
            self._resolve_frame_dir(video_id), max_frames=self.max_frames,
            slice_framepos=self.slice_framepos,
        )["video"]
        raw_frame_slice = self.rawFrameExtractor.process_raw_data(frame_data)
        raw_frame_slice = self.rawFrameExtractor.process_frame_order(
            raw_frame_slice, frame_order=self.frame_order
        )
        slice_len = min(raw_frame_slice.shape[0], self.max_frames)
        video[0, :slice_len] = raw_frame_slice[:slice_len]
        video_mask[0, :slice_len] = 1
        return video, video_mask

    def iter_prechunked_videos(self):
        """Yield ordered raw-frame chunks before any ``max_frames`` sampling.

        Each yielded tensor is padded only to ``chunk_size``. This is used by
        zero-shot chunked evaluation so a 170-frame video with chunk size 10
        produces 17 representations, rather than at most 13 representations
        after an initial 128-frame sample.
        """
        if not self.prechunk_eval:
            raise RuntimeError("iter_prechunked_videos requires chunk_size > 0 during evaluation")

        for parent_index, video_id in enumerate(self.video_to_rows):
            frame_paths = self.rawFrameExtractor._list_frame_paths(self._resolve_frame_dir(video_id))
            for start in range(0, len(frame_paths), self.chunk_size):
                chunk_paths = frame_paths[start:start + self.chunk_size]
                frame_data = self.rawFrameExtractor.get_frame_data_from_paths(chunk_paths)["video"]
                raw_frame_slice = self.rawFrameExtractor.process_raw_data(frame_data)
                raw_frame_slice = self.rawFrameExtractor.process_frame_order(
                    raw_frame_slice, frame_order=self.frame_order
                )

                video = np.zeros((1, self.chunk_size, 1, 3,
                                  self.rawFrameExtractor.size, self.rawFrameExtractor.size), dtype=np.float32)
                video_mask = np.zeros((1, self.chunk_size), dtype=np.int64)
                chunk_length = raw_frame_slice.shape[0]
                video[0, :chunk_length] = raw_frame_slice
                video_mask[0, :chunk_length] = 1
                yield parent_index, video, video_mask

    def prechunk_statistics(self):
        if not self.prechunk_eval:
            return None
        counts = []
        for video_id in self.video_to_rows:
            frame_paths = self.rawFrameExtractor._list_frame_paths(self._resolve_frame_dir(video_id))
            counts.append((len(frame_paths) + self.chunk_size - 1) // self.chunk_size)
        return {
            "videos": len(counts),
            "chunks": int(sum(counts)),
            "mean_chunks": float(np.mean(counts)) if counts else 0.0,
            "max_chunks": max(counts) if counts else 0,
        }

    def __getitem__(self, idx):
        row = self.query_rows[idx]
        pairs_text, pairs_mask, pairs_segment = self._get_text(row["caption"])
        if self.prechunk_eval:
            # Chunk visuals are loaded separately and batched in eval_epoch.
            # Keep the text dataloader light instead of loading a redundant
            # max_frames-limited parent-video tensor for every caption.
            video = np.zeros((1, 1, 1, 3,
                              self.rawFrameExtractor.size, self.rawFrameExtractor.size), dtype=np.float32)
            video_mask = np.zeros((1, 1), dtype=np.int64)
            return pairs_text, pairs_mask, pairs_segment, video, video_mask
        video, video_mask = self._get_rawframes(row["video_id"])
        return pairs_text, pairs_mask, pairs_segment, video, video_mask
