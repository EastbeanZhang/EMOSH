import numpy as np
import torch
from einops import reduce, repeat
from PIL import Image


class BasePipeline(torch.nn.Module):
    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.float16,
        height_division_factor=64,
        width_division_factor=64,
        time_division_factor=None,
        time_division_remainder=None,
    ):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.vram_management_enabled = False

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None):
        if height % self.height_division_factor != 0:
            height = (
                (height + self.height_division_factor - 1)
                // self.height_division_factor
                * self.height_division_factor
            )
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (
                (width + self.width_division_factor - 1)
                // self.width_division_factor
                * self.width_division_factor
            )
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (
                    (num_frames + self.time_division_factor - 1)
                    // self.time_division_factor
                    * self.time_division_factor
                    + self.time_division_remainder
                )
                print(
                    f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}."
                )
            return height, width, num_frames

    def preprocess_image(
        self,
        image,
        torch_dtype=None,
        device=None,
        pattern="B C H W",
        min_value=-1,
        max_value=1,
        ori_range=255,
    ):
        if isinstance(image, torch.Tensor):
            image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
            image = image * ((max_value - min_value) / ori_range) + min_value
        else:
            image = torch.Tensor(np.array(image, dtype=np.float32))
            image = image * ((max_value - min_value) / ori_range) + min_value
            image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image

    def preprocess_video(
        self,
        video,
        torch_dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
        ori_range=255,
    ):
        video = [
            self.preprocess_image(
                image,
                torch_dtype=torch_dtype,
                device=device,
                min_value=min_value,
                max_value=max_value,
                ori_range=ori_range,
            )
            for image in video
        ]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image

    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [
            self.vae_output_to_image(
                image, pattern="H W C", min_value=min_value, max_value=max_value
            )
            for image in vae_output
        ]
        return video

    def load_models_to_device(self, model_names=[]):
        if self.vram_management_enabled:
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
            torch.cuda.empty_cache()
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)

    def generate_noise(
        self,
        shape,
        seed=None,
        rand_device="cpu",
        rand_torch_dtype=torch.float32,
        device=None,
        torch_dtype=None,
        generator=None,
    ):
        if generator is None:
            generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise

    def freeze_except(self, model_names):
        for name, model in self.named_children():
            if name in model_names:
                model.train()
                model.requires_grad_(True)
            else:
                model.eval()
                model.requires_grad_(False)
