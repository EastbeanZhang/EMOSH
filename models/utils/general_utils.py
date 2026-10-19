from copy import deepcopy

import imageio
import numpy as np
import torch
from skimage import color
from torchvision.transforms.functional import InterpolationMode, resize
from tqdm import tqdm


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None, pixelformat="yuv420p"):
    writer = imageio.get_writer(
        save_path,
        fps=fps,
        quality=quality,
        ffmpeg_params=ffmpeg_params,
        macro_block_size=1,
        pixelformat=pixelformat,
    )
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


def resize_and_pad_images(image, MAX_EDGE=512, MULTIPLE=16):
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


def to_tensor_recursive(params_dict, require_grad=False, device="cuda:0"):
    """Convert nested NumPy arrays to tensors on the selected device."""
    trainable_params = {}
    for key, value in params_dict.items():
        if isinstance(value, np.ndarray):
            trainable_params[key] = torch.tensor(
                value, dtype=torch.float32, requires_grad=require_grad, device=device
            )
        elif isinstance(value, dict):
            trainable_params[key] = to_tensor_recursive(value, device=device)
        elif isinstance(value, (list, tuple)):
            new_list = []
            for item in value:
                if isinstance(item, np.ndarray):
                    new_list.append(
                        torch.tensor(
                            item, dtype=torch.float32, requires_grad=require_grad, device=device
                        )
                    )
                elif isinstance(item, dict):
                    new_list.append(to_tensor_recursive(item))
                else:
                    new_list.append(item)
            trainable_params[key] = type(value)(new_list)
        elif isinstance(value, torch.Tensor):
            trainable_params[key] = value.to(device).to(torch.float32).requires_grad_(require_grad)
        else:
            trainable_params[key] = value
    return trainable_params


def squeeze_tensor_recursive(params_dict, dim=None):
    squeezed_dict = {}
    for key, value in params_dict.items():
        if isinstance(value, torch.Tensor):
            if dim is None:
                squeezed_dict[key] = torch.squeeze(value)
            else:
                squeezed_dict[key] = torch.squeeze(value, dim=dim)
        elif isinstance(value, dict):
            squeezed_dict[key] = squeeze_tensor_recursive(value, dim=dim)
        elif isinstance(value, (list, tuple)):
            squeezed_list = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    if dim is None:
                        squeezed_list.append(torch.squeeze(item))
                    else:
                        if len(item.shape) > dim:
                            squeezed_list.append(torch.squeeze(item, dim=dim))
                        else:
                            squeezed_list.append(item)
                elif isinstance(item, dict):
                    squeezed_list.append(squeeze_tensor_recursive(item, dim=dim))
                else:
                    squeezed_list.append(item)
            squeezed_dict[key] = type(value)(squeezed_list)
        else:
            squeezed_dict[key] = value
    return squeezed_dict


def to_float32_recursive(data):
    if isinstance(data, torch.Tensor):
        return data.float()
    if isinstance(data, dict):
        return {key: to_float32_recursive(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)([to_float32_recursive(item) for item in data])
    return data


def to_device_recursive(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: to_device_recursive(value, device) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)([to_device_recursive(item, device) for item in data])
    return data


def apply_disturbance(refert_latent, refer_t_num=5, noise_pro=0.5):
    latent_frames = (refer_t_num - 1) // 4 + 1
    rand_num = torch.rand(1).item()
    sigma = torch.rand(1).item() * 0.06 + 0.02
    weight = (1 - sigma**2) ** 0.5

    if rand_num < noise_pro:
        refert_latent[:, :latent_frames] = (
            refert_latent[:, :latent_frames] * weight
            + torch.randn_like(refert_latent[:, :latent_frames]) * sigma
        )
    return refert_latent


def inputs_padding(array, target_len):
    if len(array) == 0 and target_len > 0:
        raise ValueError("Cannot pad an empty sequence.")
    if len(array) == 1:
        return [deepcopy(array[0]) for _ in range(target_len)]
    idx = 0
    flip = False
    target_array = []
    while len(target_array) < target_len:
        target_array.append(deepcopy(array[idx]))
        if flip:
            idx -= 1
        else:
            idx += 1
        if idx == 0 or idx == len(array) - 1:
            flip = not flip
    return target_array[:target_len]


def match_and_blend_colors(
    source_chunk: torch.Tensor, reference_image: torch.Tensor, strength: float
) -> torch.Tensor:
    """Blend per-channel Lab color statistics from the reference into a video chunk."""

    if strength == 0.0:
        return source_chunk

    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

    device = source_chunk.device
    dtype = source_chunk.dtype

    source_np = source_chunk.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    ref_np = reference_image.squeeze(0).squeeze(1).permute(1, 2, 0).cpu().numpy()

    source_np_01 = (source_np + 1.0) / 2.0
    ref_np_01 = (ref_np + 1.0) / 2.0

    source_np_01 = np.clip(source_np_01, 0.0, 1.0)
    ref_np_01 = np.clip(ref_np_01, 0.0, 1.0)

    try:
        ref_lab = color.rgb2lab(ref_np_01)
    except ValueError as e:
        print(
            f"Warning: Could not convert reference image to Lab: {e}. Skipping color correction for this chunk."
        )
        return source_chunk

    corrected_frames_np_01 = []
    for i in range(source_np_01.shape[0]):
        source_frame_rgb_01 = source_np_01[i]

        try:
            source_lab = color.rgb2lab(source_frame_rgb_01)
        except ValueError as e:
            print(f"Warning: Could not convert source frame {i} to Lab: {e}. Using original frame.")
            corrected_frames_np_01.append(source_frame_rgb_01)
            continue

        corrected_lab_frame = source_lab.copy()

        for j in range(3):
            mean_src, std_src = source_lab[:, :, j].mean(), source_lab[:, :, j].std()
            mean_ref, std_ref = ref_lab[:, :, j].mean(), ref_lab[:, :, j].std()

            if std_src == 0:
                corrected_lab_frame[:, :, j] = mean_ref
            else:
                corrected_lab_frame[:, :, j] = (corrected_lab_frame[:, :, j] - mean_src) * (
                    std_ref / std_src
                ) + mean_ref

        try:
            fully_corrected_frame_rgb_01 = color.lab2rgb(corrected_lab_frame)
        except ValueError as e:
            print(
                f"Warning: Could not convert corrected frame {i} back to RGB: {e}. Using original frame."
            )
            corrected_frames_np_01.append(source_frame_rgb_01)
            continue

        fully_corrected_frame_rgb_01 = np.clip(fully_corrected_frame_rgb_01, 0.0, 1.0)

        blended_frame_rgb_01 = (
            1 - strength
        ) * source_frame_rgb_01 + strength * fully_corrected_frame_rgb_01
        corrected_frames_np_01.append(blended_frame_rgb_01)

    corrected_chunk_np_01 = np.stack(corrected_frames_np_01, axis=0)

    corrected_chunk_np_minus1_1 = (corrected_chunk_np_01 * 2.0) - 1.0

    corrected_chunk_tensor = (
        torch.from_numpy(corrected_chunk_np_minus1_1).permute(3, 0, 1, 2).unsqueeze(0)
    )
    corrected_chunk_tensor = corrected_chunk_tensor.contiguous()
    output_tensor = corrected_chunk_tensor.to(device=device, dtype=dtype)
    return output_tensor
