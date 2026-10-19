import math

import torch
import torch.nn.functional as F


def canvas_size(width, height, size=None, resolution_area=None):
    if min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive.")
    if size is not None:
        if resolution_area is not None:
            raise ValueError("Choose a fixed canvas or an area budget, not both.")
        if min(size) < 64 or any(value % 16 for value in size):
            raise ValueError("Canvas dimensions must be multiples of 16 and at least 64.")
        return tuple(size)
    if resolution_area is None:
        return tuple(max(64, (value + 15) // 16 * 16) for value in (width, height))
    if min(resolution_area) <= 0 or math.prod(resolution_area) < 64 * 64:
        raise ValueError("The resolution area must be positive and at least 64 x 64.")
    area = math.prod(resolution_area)
    scale = math.sqrt(area / (width * height))
    new_width = max(64, int(width * scale) // 16 * 16)
    new_height = max(64, int(height * scale) // 16 * 16)
    if new_width * new_height > area:
        if width >= height:
            new_width = area // new_height // 16 * 16
        else:
            new_height = area // new_width // 16 * 16
    return new_width, new_height


def fit_canvas(image, tracking, size, pad_only=False, content_size=None):
    """Apply the same aspect-preserving pixel transform to the image and camera."""
    height, width = image.shape[-2:]
    target_width, target_height = size
    if (width, height) == size and (content_size is None or content_size == size):
        mask = torch.ones_like(image)
        return image, mask
    content_width, content_height = content_size or size
    scale = 1.0 if pad_only else min(content_width / width, content_height / height)
    tx = (content_width - scale * width) / 2 + (target_width - content_width) // 2
    ty = (content_height - scale * height) / 2 + (target_height - content_height) // 2
    if pad_only:
        tx, ty = math.floor(tx), math.floor(ty)
    transform = image.new_tensor([[scale, 0, tx], [0, scale, ty], [0, 0, 1]])
    y, x = torch.meshgrid(
        torch.arange(target_height, dtype=image.dtype, device=image.device),
        torch.arange(target_width, dtype=image.dtype, device=image.device),
        indexing="ij",
    )
    grid = torch.stack(
        (2 * ((x - tx) / scale + 0.5) / width - 1, 2 * ((y - ty) / scale + 0.5) / height - 1),
        dim=-1,
    )[None]
    fitted = F.grid_sample(image[None], grid, align_corners=False)[0]
    mask = (F.grid_sample(torch.ones_like(image)[None], grid, align_corners=False)[0] > 0).to(
        image.dtype
    )
    camera = tracking["smplx_params"]
    intrinsic = camera.get("K_intrinsic")
    if intrinsic is None:
        intrinsic = image.new_tensor(
            [
                [camera["focal_px"].item(), 0, width / 2],
                [0, camera["focal_py"].item(), height / 2],
                [0, 0, 1],
            ]
        )
    camera["K_intrinsic"] = transform @ intrinsic
    camera["focal_px"] = camera["focal_px"] * scale
    camera["focal_py"] = camera["focal_py"] * scale
    crop = tracking["full_crop"]
    crop["M_o2c-full"] = transform @ crop["M_o2c-full"]
    crop["M_c2o-full"] = torch.linalg.inv(crop["M_o2c-full"])
    return fitted, mask


def add_resolution_arguments(parser):
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--resolution-area",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Aspect-preserving pixel area budget, for example 1280 720.",
    )


def validate_resolution_arguments(parser, args):
    if (args.width is None) != (args.height is None):
        parser.error("Provide both --width and --height, or omit both.")
    size = None if args.width is None else (args.width, args.height)
    try:
        canvas_size(512, 512, size=size, resolution_area=args.resolution_area)
    except ValueError as error:
        parser.error(str(error))
