import os
import random
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode, resize

from .dataset_index import DatasetIndex
from .general_utils import (
    resize_and_pad_images,
    squeeze_tensor_recursive,
    to_tensor_recursive,
)
from .graphics_utils import get_full_proj_matrix
from .image_geometry import canvas_size, fit_canvas
from .lmdb import LMDBEngine


class Tracked_VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split="train",
        base_path=None,
        metadata_path=None,
        num_frames=77,
        time_division_factor=4,
        time_division_remainder=1,
        max_pixels=512 * 512,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        resolution_area=None,
        cache_size=8,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            repeat = args.dataset_repeat
        self.split = split
        assert self.split in ["train", "valid", "test"], f"Invalid split: {self.split}"
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        self.cache_size = cache_size
        self._cache_pid = os.getpid()
        self.tracked_data = OrderedDict()
        self.tracked_data_id_share = {}
        self.lmdb_engines = OrderedDict()
        self.videos_promt = {}

        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        if (width is None) != (height is None):
            raise ValueError("Provide both width and height, or omit both.")
        self.fixed_size = None if width is None else (width, height)
        self.resolution_area = resolution_area
        canvas_size(512, 512, self.fixed_size, resolution_area)
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat

        self.dynamic_resolution = self.fixed_size is None

        self.videos_info = DatasetIndex(self.base_path, cache_size=cache_size)
        self.videos_ids = list(self.videos_info.iter_ids(self.split))
        if not self.videos_ids:
            raise ValueError(f"No videos in {self.split} split")
        _, self.frames_num, self.minimum_frames = self.videos_info.summary(self.split)

    def __getitem__(self, data_id):
        data_id = data_id % (len(self.videos_ids))
        video_id = self.videos_ids[data_id]
        return self._load_one_record(video_id)

    def __len__(self):
        return len(self.videos_ids) * self.repeat

    def slice(self, slice, start=0):
        if start + slice > len(self.videos_ids):
            self.videos_ids = self.videos_ids[:slice]
        else:
            self.videos_ids = self.videos_ids[start : start + slice]

    def shuffle_slice(self, slice_num):
        slice_num = slice_num % len(self.videos_ids)
        random.seed(time.time())
        random.shuffle(self.videos_ids)
        self.videos_ids = self.videos_ids[:slice_num]

    def _get_lmdb_engine(self, video_id):
        self._ensure_process()
        lmdb_path = os.path.abspath(os.path.join(self._get_video_dir(video_id), "image_lmdb"))
        if lmdb_path not in self.lmdb_engines:
            if len(self.lmdb_engines) >= self.cache_size:
                _, engine = self.lmdb_engines.popitem(last=False)
                engine.close()
            self.lmdb_engines[lmdb_path] = LMDBEngine(lmdb_path, write=False)
        self.lmdb_engines.move_to_end(lmdb_path)
        return self.lmdb_engines[lmdb_path]

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.shape[-1], image.shape[-2]
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_num_frames(self, num_frames):
        while (
            num_frames > 1
            and num_frames % self.time_division_factor != self.time_division_remainder
        ):
            num_frames -= 1
        return num_frames

    def _choose_image(self, video_id, frames_keys):
        if self.split == "train":
            source_key = random.sample(frames_keys, k=1)[0]
        else:
            source_key = frames_keys[0]

        return source_key

    def _select_canvas(self, video_id):
        record = self.videos_info[video_id]
        width, height = record.get("width"), record.get("height")
        if width is None or height is None:
            image = self._get_lmdb_engine(video_id)[f"{record['frames_keys'][0]}/full_crop_image"]
            height, width = image.shape[-2:]
        self.width, self.height = canvas_size(width, height, self.fixed_size, self.resolution_area)
        self._input_size = (width, height)

    def _load_one_record(self, video_id):
        self._select_canvas(video_id)
        one_record = {}
        frame_keys = self.videos_info[video_id]["frames_keys"]
        source_key = self._choose_image(video_id, frame_keys)
        _lmdb_engine = self._get_lmdb_engine(video_id)

        self._load_tracked_file(video_id)

        target_infos = []
        if len(frame_keys) > self.num_frames:
            target_start_idx = random.sample(range(len(frame_keys) - self.num_frames + 1), k=1)[0]
            target_keys = frame_keys[target_start_idx : target_start_idx + self.num_frames]
        else:
            num_frames = self.get_num_frames(len(frame_keys))
            target_keys = frame_keys[:num_frames]

        source_tracking_info, source_image, source_mask = self._load_one_info(
            _lmdb_engine, video_id, source_key, load_box=True
        )
        source_image = torchvision.transforms.functional.resize(
            source_image, (self.height, self.width), antialias=True
        )
        x1, y1, x2, y2 = source_tracking_info["head_box"]
        head_image = source_image.squeeze(0).permute(1, 2, 0)[y1:y2, x1:x2, :]
        head_image = resize_and_pad_images(head_image)

        target_images = []
        for target_key in target_keys:
            traget_tracking_info, target_image, target_mask = self._load_one_info(
                _lmdb_engine, video_id, target_key
            )
            target_image = self.crop_and_resize(target_image, self.height, self.width)
            target_mask = self.crop_and_resize(target_mask, self.height, self.width)
            focal_px, focal_py = (
                traget_tracking_info["smplx_params"]["focal_px"],
                traget_tracking_info["smplx_params"]["focal_py"],
            )
            tanHalfFov = torch.tensor(
                [self.width / (focal_px * 2), self.height / (focal_py * 2)], dtype=torch.float32
            )
            view_matrix, full_proj_matrix = get_full_proj_matrix(
                traget_tracking_info["w2c_cam"],
                tanHalfFov,
                intrinsics=traget_tracking_info["smplx_params"].get("K_intrinsic"),
                image_size=(self.width, self.height),
            )
            traget_tracking_info["render_cam_params"] = {
                "world_view_transform": view_matrix,
                "full_proj_transform": full_proj_matrix,
                "tanfovx": tanHalfFov[0],
                "tanfovy": tanHalfFov[1],
                "image_height": self.height,
                "image_width": self.width,
                "camera_center": traget_tracking_info["c2w_cam"][:3, 3],
            }
            target_images.append(target_image.squeeze(0).permute(1, 2, 0))
            target_infos.append(traget_tracking_info)
        one_record["source_info"] = source_tracking_info
        one_record["source_info"].update({"image": source_image * source_mask})
        one_record["source_info"] = add_batch_dim_recursive(one_record["source_info"])
        one_record["source_image"] = source_image.squeeze(0).permute(1, 2, 0)
        one_record["head_image"] = head_image
        one_record["target_infos"] = add_batch_dim_recursive(target_infos)

        one_record["target_video"] = target_images
        one_record["prompt"] = self.videos_promt[video_id]
        one_record["video_id"] = video_id
        return one_record

    def _load_one_info(self, _lmdb_engine, video_id, frame_key, load_box=False):
        _image = _lmdb_engine[f"{frame_key}/full_crop_image"].float() / 255.0
        tracking_info = deepcopy(self.tracked_data[video_id][frame_key])
        tracking_info["id_params"] = deepcopy(self.tracked_data_id_share[video_id])

        tracking_info = to_tensor_recursive(tracking_info, device="cpu")
        tracking_info = squeeze_tensor_recursive(tracking_info, dim=0)
        reference_size = getattr(self, "_reference_size", None)
        _image, _mask = fit_canvas(
            _image,
            tracking_info,
            (self.width, self.height),
            pad_only=self.fixed_size is None
            and self.resolution_area is None
            and (reference_size is None or reference_size == (_image.shape[-1], _image.shape[-2])),
            content_size=reference_size,
        )

        RT = tracking_info["smplx_params"]["w2c_RT"]
        c2c_mat = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )

        RT_mat = RT
        w2c_cam = torch.matmul(c2c_mat, RT_mat)
        c2w_cam = torch.linalg.inv(w2c_cam)
        tracking_info["w2c_cam"], tracking_info["c2w_cam"] = w2c_cam, c2w_cam
        if load_box:
            (
                tracking_info["head_box"],
                tracking_info["left_hand_box"],
                tracking_info["right_hand_box"],
            ) = self._load_box(tracking_info)

        return tracking_info, _image, _mask

    def _load_tracked_file(self, video_id):
        self._ensure_process()
        video_dir = self._get_video_dir(video_id)
        if video_id not in self.tracked_data.keys():
            parameters = np.load(
                os.path.join(video_dir, "optim_tracking.npy"), allow_pickle=True
            ).item()
            identity = np.load(
                os.path.join(video_dir, "id_share_params.npy"), allow_pickle=True
            ).item()
            text_dir = os.path.join(video_dir, "prompt.txt")
            prompt = "视频中的人在做动作"
            if os.path.exists(text_dir):
                with open(text_dir) as stream:
                    prompt = stream.read().strip()
            if len(self.tracked_data) >= self.cache_size:
                old_id, _ = self.tracked_data.popitem(last=False)
                self.tracked_data_id_share.pop(old_id)
                self.videos_promt.pop(old_id)
            self.tracked_data[video_id] = parameters
            self.tracked_data_id_share[video_id] = identity
            self.videos_promt[video_id] = prompt
        self.tracked_data.move_to_end(video_id)

    def _get_video_dir(self, video_id):
        if "video_dir" in self.videos_info[video_id]:
            video_dir = os.path.join(self.base_path, self.videos_info[video_id]["video_dir"])
        else:
            video_dir = self.base_path
        return video_dir

    def _load_box(self, tracking_info):
        crop_info = {
            "full_crop": tracking_info["full_crop"],
            "head_crop": tracking_info["head_crop"],
            "left_hand_crop": tracking_info["left_hand_crop"],
            "right_hand_crop": tracking_info["right_hand_crop"],
        }

        scale = 1.0
        image_size = [self.width, self.height]
        head_crop_size = 256
        hand_crop_size = 256

        head_box_o = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [head_crop_size, 0.0, 1.0],
                [0.0, head_crop_size, 1.0],
                [head_crop_size, head_crop_size, 1.0],
            ]
        )
        hand_box_o = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [hand_crop_size, 0.0, 1.0],
                [0.0, hand_crop_size, 1.0],
                [hand_crop_size, hand_crop_size, 1.0],
            ]
        )

        body_crop = crop_info["full_crop"]
        head_crop = crop_info["head_crop"]
        left_hand_crop = crop_info["left_hand_crop"]
        right_hand_crop = crop_info["right_hand_crop"]

        head_crop["M_c2o"] = torch.nan_to_num(head_crop["M_c2o"])
        left_hand_crop["M_c2o"] = torch.nan_to_num(left_hand_crop["M_c2o"])
        right_hand_crop["M_c2o"] = torch.nan_to_num(right_hand_crop["M_c2o"])

        head_box = body_crop["M_o2c-full"] @ head_crop["M_c2o"] @ head_box_o[:, :, None]
        left_hand_box = body_crop["M_o2c-full"] @ left_hand_crop["M_c2o"] @ hand_box_o[:, :, None]
        right_hand_box = body_crop["M_o2c-full"] @ right_hand_crop["M_c2o"] @ hand_box_o[:, :, None]

        head_box *= scale
        left_hand_box *= scale
        right_hand_box *= scale
        head_box[:, 0] = head_box[:, 0].clamp(0, image_size[0] - 1)
        head_box[:, 1] = head_box[:, 1].clamp(0, image_size[1] - 1)
        left_hand_box[:, 0] = left_hand_box[:, 0].clamp(0, image_size[0] - 1)
        left_hand_box[:, 1] = left_hand_box[:, 1].clamp(0, image_size[1] - 1)
        right_hand_box[:, 0] = right_hand_box[:, 0].clamp(0, image_size[0] - 1)
        right_hand_box[:, 1] = right_hand_box[:, 1].clamp(0, image_size[1] - 1)

        head_left, head_right = int(head_box.min(dim=0)[0][0]), int(head_box.max(dim=0)[0][0])
        head_top, head_bottom = int(head_box.min(dim=0)[0][1]), int(head_box.max(dim=0)[0][1])

        left_hand_left, left_hand_right = (
            int(left_hand_box.min(dim=0)[0][0]),
            int(left_hand_box.max(dim=0)[0][0]),
        )
        left_hand_top, left_hand_bottom = (
            int(left_hand_box.min(dim=0)[0][1]),
            int(left_hand_box.max(dim=0)[0][1]),
        )
        right_hand_left, right_hand_right = (
            int(right_hand_box.min(dim=0)[0][0]),
            int(right_hand_box.max(dim=0)[0][0]),
        )
        right_hand_top, right_hand_bottom = (
            int(right_hand_box.min(dim=0)[0][1]),
            int(right_hand_box.max(dim=0)[0][1]),
        )
        head_box = torch.tensor([head_left, head_top, head_right, head_bottom], dtype=torch.long)
        left_hand_box = torch.tensor(
            [left_hand_left, left_hand_top, left_hand_right, left_hand_bottom], dtype=torch.long
        )
        right_hand_box = torch.tensor(
            [right_hand_left, right_hand_top, right_hand_right, right_hand_bottom], dtype=torch.long
        )
        if head_box[2] <= head_box[0] or head_box[3] <= head_box[1]:
            head_box = torch.tensor([0, 0, image_size[0] - 1, image_size[1] - 1], dtype=torch.long)
        if right_hand_box[2] <= right_hand_box[0] or right_hand_box[3] <= right_hand_box[1]:
            right_hand_box = torch.tensor(
                [0, 0, image_size[0] - 1, image_size[1] - 1], dtype=torch.long
            )
        if left_hand_box[2] <= left_hand_box[0] or left_hand_box[3] <= left_hand_box[1]:
            left_hand_box = torch.tensor(
                [0, 0, image_size[0] - 1, image_size[1] - 1], dtype=torch.long
            )

        return head_box, left_hand_box, right_hand_box

    def _resize_pad_images(self, image, MAX_EDGE=512, MULTIPLE=16):
        H, W, C = image.shape
        if H < 10 or W < 10:
            return torch.zeros((MAX_EDGE, MAX_EDGE, C), dtype=image.dtype)
        longest_edge = max(H, W)
        scale_factor = MAX_EDGE / longest_edge
        new_H = int(H * scale_factor)
        new_W = int(W * scale_factor)
        image_chw = image.permute(2, 0, 1).contiguous()
        resized_image_chw = resize(
            image_chw, size=(new_H, new_W), interpolation=InterpolationMode.BILINEAR
        )
        padded_H = (new_H + MULTIPLE - 1) // MULTIPLE * MULTIPLE
        padded_W = (new_W + MULTIPLE - 1) // MULTIPLE * MULTIPLE
        pad_H_total = padded_H - new_H
        pad_W_total = padded_W - new_W
        pad_left = pad_W_total // 2
        pad_right = pad_W_total - pad_left
        pad_top = pad_H_total // 2
        pad_bottom = pad_H_total - pad_top
        padding_dims = (pad_left, pad_right, pad_top, pad_bottom)
        padded_image_chw = torch.nn.functional.pad(
            resized_image_chw, padding_dims, mode="constant", value=0.0
        )
        final_image = padded_image_chw.permute(1, 2, 0).contiguous()
        return final_image

    def close_all_engines(self):
        for engine in self.lmdb_engines.values():
            engine.close()
        self.lmdb_engines.clear()
        self.tracked_data.clear()
        self.tracked_data_id_share.clear()
        self.videos_promt.clear()
        if hasattr(self, "videos_info"):
            self.videos_info.close()

    def _ensure_process(self):
        if self._cache_pid != os.getpid():
            self.close_all_engines()
            self._cache_pid = os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(
            tracked_data=OrderedDict(),
            tracked_data_id_share={},
            lmdb_engines=OrderedDict(),
            videos_promt={},
            _cache_pid=None,
        )
        return state

    def __del__(self):
        if hasattr(self, "lmdb_engines"):
            self.close_all_engines()


class Tracked_VideoDataset_infer(Tracked_VideoDataset):
    def _load_source_info(self, video_id, load_box=True):
        self._select_canvas(video_id)
        self._reference_size = (
            self._input_size if self.fixed_size is None and self.resolution_area is None else None
        )
        self._load_tracked_file(video_id)
        _lmdb_engine = self._get_lmdb_engine(video_id)
        frame_keys = self.videos_info[video_id]["frames_keys"]
        frame_key = frame_keys[0]
        source_tracking_info, source_image, source_mask = self._load_one_info(
            _lmdb_engine, video_id, frame_key, load_box=load_box
        )
        source_image = torchvision.transforms.functional.resize(
            source_image, (self.height, self.width), antialias=True
        )
        x1, y1, x2, y2 = source_tracking_info["head_box"]
        head_image = source_image.squeeze(0).permute(1, 2, 0)[y1:y2, x1:x2, :]
        head_image = resize_and_pad_images(head_image)

        info_dict = {}
        info_dict["source_info"] = source_tracking_info
        info_dict["source_info"].update({"image": source_image * source_mask})
        info_dict["source_info"] = add_batch_dim_recursive(info_dict["source_info"])
        info_dict["source_image"] = source_image.squeeze(0).permute(1, 2, 0)
        info_dict["head_image"] = head_image
        info_dict["video_id"] = video_id
        return info_dict

    def _load_target_info(self, video_id):
        if self.width is None or self.height is None:
            self._select_canvas(video_id)
        self._load_tracked_file(video_id)
        _lmdb_engine = self._get_lmdb_engine(video_id)
        frame_keys = self.videos_info[video_id]["frames_keys"]
        target_infos = []
        target_images = []
        for target_key in frame_keys:
            traget_tracking_info, target_image, target_mask = self._load_one_info(
                _lmdb_engine, video_id, target_key
            )
            target_image = self.crop_and_resize(target_image, self.height, self.width)
            target_mask = self.crop_and_resize(target_mask, self.height, self.width)
            focal_px, focal_py = (
                traget_tracking_info["smplx_params"]["focal_px"],
                traget_tracking_info["smplx_params"]["focal_py"],
            )
            tanHalfFov = torch.tensor(
                [self.width / (focal_px * 2), self.height / (focal_py * 2)], dtype=torch.float32
            )
            view_matrix, full_proj_matrix = get_full_proj_matrix(
                traget_tracking_info["w2c_cam"],
                tanHalfFov,
                intrinsics=traget_tracking_info["smplx_params"].get("K_intrinsic"),
                image_size=(self.width, self.height),
            )
            traget_tracking_info["render_cam_params"] = {
                "world_view_transform": view_matrix,
                "full_proj_transform": full_proj_matrix,
                "tanfovx": tanHalfFov[0],
                "tanfovy": tanHalfFov[1],
                "image_height": self.height,
                "image_width": self.width,
                "camera_center": traget_tracking_info["c2w_cam"][:3, 3],
            }
            target_images.append(target_image.squeeze(0).permute(1, 2, 0))
            target_infos.append(traget_tracking_info)
        info_dict = {}
        info_dict["target_infos"] = add_batch_dim_recursive(target_infos)
        info_dict["target_video"] = target_images
        info_dict["prompt"] = self.videos_promt[video_id]
        info_dict["video_id"] = video_id
        return info_dict

    def __getitem__(self, data_id):
        if data_id >= len(self):
            raise IndexError("Index out of bounds of the repeated dataset.")
        video_id = self.videos_ids[data_id]
        return self._load_one_record(video_id)

    def __len__(self):
        return len(self.videos_ids)

    def get_num_frames(self, num_frames):
        return num_frames


def add_batch_dim_recursive(data):
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(0)

    elif isinstance(data, dict):
        return {key: add_batch_dim_recursive(value) for key, value in data.items()}

    elif isinstance(data, (list, tuple)):
        return type(data)([add_batch_dim_recursive(item) for item in data])

    else:
        return [data]
