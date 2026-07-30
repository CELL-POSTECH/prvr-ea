import torch as th
import numpy as np
from PIL import Image
import os
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2

class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2


class RawFrameExtractor(RawVideoExtractorCV2):
    """Apply CLIP's original image preprocessing to decoded frame files."""

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def _list_frame_paths(self, frame_dir):
        if not os.path.isdir(frame_dir):
            raise FileNotFoundError("frame directory not found: {}".format(frame_dir))
        frame_paths = [
            os.path.join(frame_dir, name)
            for name in os.listdir(frame_dir)
            if name.lower().endswith(self.IMAGE_EXTENSIONS)
        ]
        frame_paths.sort()
        if not frame_paths:
            raise FileNotFoundError("no frame images in directory: {}".format(frame_dir))
        return frame_paths

    @staticmethod
    def _sample_frame_paths(frame_paths, max_frames, slice_framepos=0):
        if max_frames is None or max_frames <= 0 or len(frame_paths) <= max_frames:
            return frame_paths
        if slice_framepos == 0:
            return frame_paths[:max_frames]
        if slice_framepos == 1:
            return frame_paths[-max_frames:]
        sample_idx = np.linspace(0, len(frame_paths) - 1, num=max_frames, dtype=int)
        return [frame_paths[idx] for idx in sample_idx]

    def get_frame_data_from_paths(self, frame_paths):
        """Load and preprocess an already selected ordered frame sequence."""
        if not frame_paths:
            raise ValueError("frame_paths must not be empty")
        images = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                images.append(self.transform(image.convert("RGB")))
        return {"video": th.stack(images, dim=0)}

    def get_frame_data(self, frame_dir, max_frames=None, slice_framepos=0):
        frame_paths = self._sample_frame_paths(
            self._list_frame_paths(frame_dir), max_frames=max_frames, slice_framepos=slice_framepos
        )
        return self.get_frame_data_from_paths(frame_paths)
