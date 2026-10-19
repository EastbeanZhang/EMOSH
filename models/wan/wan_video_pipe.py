# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..human_model.ehm import EHM
from ..human_model.render_utils import (
    Mesh_Render,
    draw_135_MP_keypoints,
    perspective_projection,
)
from ..utils.augmentation import apply_transform
from ..utils.general_utils import (
    apply_disturbance,
    inputs_padding,
    match_and_blend_colors,
    resize_and_pad_images,
    to_device_recursive,
    to_float32_recursive,
)
from .distributed.util import gather_forward, get_rank, get_world_size
from .lora import GeneralLoRALoader
from .modules.animate_adapter import Extra_Animate_Adapter
from .modules.animate_utils import TensorList
from .modules.clip import CLIPModel
from .modules.model import (
    sinusoidal_embedding_1d,
)
from .modules.model_animate import WanAnimateModel
from .modules.t5 import T5EncoderModel
from .modules.utils import load_state_dict
from .modules.vae2_1 import Wan2_1_VAE
from .schedulers.flow_match import FlowMatchScheduler
from .schedulers.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils import BasePipeline


class WanVideoPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )

        self.text_encoder: T5EncoderModel = None
        self.clip: CLIPModel = None
        self.dit: WanAnimateModel = None
        self.vae: Wan2_1_VAE = None

        self.extra_animate_adapter: Extra_Animate_Adapter = None
        self.in_iteration_models = ("dit", "extra_animate_adapter")
        self.model_fn = model_fn_wan_video

    def load_lora_extra(self, checkpoint):
        state = load_state_dict(checkpoint, torch_dtype=self.torch_dtype, device="cpu")
        control = {
            key.removeprefix("extra_animate_adapter."): value
            for key, value in state.items()
            if key.startswith("extra_animate_adapter.")
        }
        lora = {
            key.removeprefix("dit."): value
            for key, value in state.items()
            if key.startswith("dit.")
        }
        if not control or not lora:
            raise ValueError("Checkpoint must contain EMOSH control and LoRA parameters.")
        # Research checkpoints also contain an unused position embedding.
        for key in ["rendering_patch_pos_embedding.weight", "rendering_patch_pos_embedding.bias"]:
            control.pop(key, None)
        self.extra_animate_adapter.load_state_dict(control, strict=True)
        GeneralLoRALoader(device=self.device, torch_dtype=self.torch_dtype).load(self.dit, lora)

    def training_loss(self, **inputs):
        max_timestep_boundary = int(1 * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(0 * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(
            dtype=self.torch_dtype, device=self.device
        )
        input_latents = inputs["input_latents"][0].unsqueeze(0)
        noise = inputs["noise"][0].unsqueeze(0)
        latents = self.scheduler.add_noise(input_latents, noise, timestep)
        training_target = self.scheduler.training_target(input_latents, noise, timestep)

        with torch.autocast(
            device_type=torch.device(self.device).type, dtype=self.torch_dtype, enabled=True
        ):
            noise_pred = TensorList(
                self.model_fn(
                    **inputs,
                    x=TensorList([latents[0].to(self.torch_dtype).to(self.device)]),
                    t=timestep,
                )
            )
        noise_pred = noise_pred[0].unsqueeze(0)
        loss = torch.nn.functional.mse_loss(
            noise_pred[:, :, 1:].float(), training_target[:, :, 1:].float()
        )
        loss = loss * self.scheduler.training_weight(timestep)
        return loss

    def enable_vram_management(self):
        self.vram_management_enabled = True

    @staticmethod
    def from_pretrained(
        config,
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device="cuda",
        offload_device=None,
        convert_model_dtype=False,
        use_relighting_lora=False,
        assets="assets",
    ):
        if use_relighting_lora:
            raise ValueError("Relighting is not part of the EMOSH training pipeline.")
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.scheduler = FlowMatchScheduler(
            shift=5,
            sigma_min=0,
            extra_one_step=True,
            num_train_timesteps=config.num_train_timesteps,
        )
        pipe.num_train_timesteps = config.num_train_timesteps
        pipe.param_dtype = torch_dtype
        encoder_device = offload_device or device
        pipe.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=encoder_device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )
        pipe.clip = CLIPModel(
            dtype=torch.float16,
            device=encoder_device,
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer),
        )
        pipe.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint), device=encoder_device
        )
        if convert_model_dtype:
            pipe.vae.model.to(torch_dtype)
            pipe.vae.dtype = torch_dtype
        pipe.dit = WanAnimateModel.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch_dtype,
            device_map=device,
            local_files_only=True,
            token=False,
        )
        pipe.dit.eval().requires_grad_(False)
        pipe.extra_animate_adapter = Extra_Animate_Adapter(
            dim=pipe.dit.dim,
            patch_size=pipe.dit.patch_size,
        ).to(device=encoder_device, dtype=torch_dtype)
        pipe.extra_animate_adapter.init_weights(pipe.dit.blocks, pipe.dit.pose_patch_embedding)
        pipe.height_division_factor = config.vae_stride[1] * config.patch_size[1]
        pipe.width_division_factor = config.vae_stride[2] * config.patch_size[2]
        pipe.ehm = EHM(assets).to(encoder_device)
        pipe.mesh_renderer = Mesh_Render(pipe.ehm.smplx.faces_tensor)
        vertices = pipe.ehm.smplx.v_template.clone()
        low, high = vertices.amin(0, keepdim=True), vertices.amax(0, keepdim=True)
        pipe.vertices_color = (vertices - low) / (high - low + 1e-6)
        pipe.sp_size = 1
        pipe.sample_neg_prompt = config.sample_neg_prompt
        pipe.sample_prompt = config.prompt
        return pipe

    def get_valid_len(self, real_len, clip_len=81, overlap=1):
        real_clip_len = clip_len - overlap
        last_clip_num = (real_len - overlap) % real_clip_len
        if last_clip_num == 0:
            extra = 0
        else:
            extra = real_clip_len - last_clip_num
        target_len = real_len + extra
        return target_len

    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def padding_resize(
        self,
        img_ori,
        height=512,
        width=512,
        padding_color=(0, 0, 0),
        interpolation=cv2.INTER_LINEAR,
    ):
        ori_height = img_ori.shape[0]
        ori_width = img_ori.shape[1]
        channel = img_ori.shape[2]

        img_pad = np.zeros((height, width, channel))
        if channel == 1:
            img_pad[:, :, 0] = padding_color[0]
        else:
            img_pad[:, :, 0] = padding_color[0]
            img_pad[:, :, 1] = padding_color[1]
            img_pad[:, :, 2] = padding_color[2]

        if (ori_height / ori_width) > (height / width):
            new_width = int(height / ori_height * ori_width)
            img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
            padding = int((width - new_width) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            img_pad[:, padding : padding + new_width, :] = img
        else:
            new_height = int(width / ori_width * ori_height)
            img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
            padding = int((height - new_height) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            img_pad[padding : padding + new_height, :, :] = img

        img_pad = np.uint8(img_pad)
        return img_pad

    def padding_resize_tensor(
        self, img_tensor, height=512, width=512, padding_color=(0, 0, 0), interpolation="bilinear"
    ):
        B, C, H_ori, W_ori = img_tensor.shape
        device = img_tensor.device
        dtype = img_tensor.dtype
        target_aspect_ratio = height / width
        original_aspect_ratio = H_ori / W_ori

        if original_aspect_ratio > target_aspect_ratio:
            new_height = height
            new_width = int(height / H_ori * W_ori)
        else:
            new_width = width
            new_height = int(width / W_ori * H_ori)

        img_resized = F.interpolate(
            img_tensor,
            size=(new_height, new_width),
            mode=interpolation,
            align_corners=False if interpolation in ("bilinear", "bicubic") else None,
        )

        color_tensor = torch.as_tensor(padding_color, device=device, dtype=dtype)
        if C == 1 and color_tensor.ndim == 1 and color_tensor.shape[0] >= 1:
            color_tensor = color_tensor[0]
        elif C == 3 and color_tensor.ndim == 1 and color_tensor.shape[0] >= 3:
            color_tensor = color_tensor[:3].view(C, 1, 1)

        img_pad = torch.ones((B, C, height, width), device=device, dtype=dtype) * color_tensor
        pad_top = (height - new_height) // 2
        pad_bottom = height - new_height - pad_top
        pad_left = (width - new_width) // 2
        pad_right = width - new_width - pad_left
        img_pad[:, :, pad_top : height - pad_bottom, pad_left : width - pad_right] = img_resized
        return img_pad

    def process(self, inputs_posi, inputs_nega, inputs_shared, training=False):
        clip_len = inputs_shared["clip_len"]
        seed_g = inputs_shared["seed_g"]
        cfg_scale = inputs_shared["cfg_scale"]
        refer_t_num = inputs_shared.get("refer_t_num", 0)

        with torch.no_grad():
            inputs_shared = self.process_mesh_rendering(inputs_shared)

        enhance_pro = 0.7 if refer_t_num == 0 else 0.5
        enhancing = torch.rand(1).item() < enhance_pro
        if training and enhancing:
            input_image, input_mask, input_rendering = apply_transform(
                inputs_shared["source_image"].cpu().numpy(),
                inputs_shared["source_rendering_image"].float().cpu().numpy(),
            )
            inputs_shared["source_image"] = torch.tensor(
                input_image, dtype=torch.float32, device=self.device
            )
            input_mask = torch.tensor(input_mask, dtype=torch.float32, device=self.device)
            inputs_shared["source_rendering_image"] = torch.tensor(
                input_rendering, dtype=torch.float32, device=self.device
            )
        else:
            input_mask = torch.ones(
                (inputs_shared["source_image"].shape[0], inputs_shared["source_image"].shape[1], 1),
                dtype=self.torch_dtype,
                device=self.device,
            )
        inputs_shared = self.process_rendering_vae(inputs_shared)

        with torch.autocast(
            device_type=torch.device(self.device).type, dtype=self.torch_dtype, enabled=True
        ):
            if "context" not in inputs_posi and "context_null" not in inputs_nega:
                self.load_models_to_device(["text_encoder"])
                context = self.text_encoder([inputs_posi["prompt"]], self.device)
                context_null = None
                if cfg_scale > 1.0:
                    context_null = self.text_encoder([inputs_nega["prompt"]], self.device)
                inputs_posi["context"] = context
                inputs_nega["context"] = context_null

            self.load_models_to_device(["vae"])
            refer_image = self.preprocess_image(
                inputs_shared["source_image"], ori_range=1.0, torch_dtype=torch.float32
            )[:, :3]
            height, width = refer_image.shape[-2:]
            refer_image = self.padding_resize_tensor(refer_image, height=height, width=width)

            B, _, H, W = refer_image.shape
            T = clip_len
            lat_h = H // 8
            lat_w = W // 8
            lat_t = T // 4 + 1

            ref_latents = self.vae.encode(
                refer_image[:, :, None].to(self.device).to(self.torch_dtype)
            )
            ref_latents = torch.stack(ref_latents)
            input_mask = torch.nn.functional.interpolate(
                input_mask.permute(2, 0, 1).unsqueeze(0),
                size=(lat_h, lat_w),
                mode="bilinear",
                align_corners=False,
            )
            mask_ref = self.get_i2v_mask(
                1, lat_h, lat_w, 0, mask_pixel_values=input_mask, device=self.device
            )

            y_ref = torch.concat([mask_ref, ref_latents[0]]).to(
                dtype=self.torch_dtype, device=self.device
            )
            if inputs_shared.get("refer_new_latent", None) is not None and inputs_shared.get(
                "add_refer", False
            ):
                y_ref_2 = torch.concat([mask_ref, inputs_shared["refer_new_latent"][0]]).to(
                    dtype=self.torch_dtype, device=self.device
                )
                y_ref = torch.cat([y_ref, y_ref_2], dim=1)
                inputs_shared["extra_refer_num"] = 1

            if training:
                input_video = self.preprocess_video(inputs_shared["input_video"], ori_range=1.0)
                input_latents = self.vae.encode(input_video.to(self.device).to(self.torch_dtype))

                input_latents = torch.cat([ref_latents[0], input_latents[0]], dim=1)
                inputs_shared["input_latents"] = [input_latents]
                inputs_shared["refer_temporal_image"] = input_video[0, :, :refer_t_num]

            refer_temporal_image = inputs_shared.get("refer_temporal_image", None)
            if refer_temporal_image is not None and refer_t_num > 0:
                refer_temporal_image = torch.nn.functional.interpolate(
                    refer_temporal_image[:, :refer_t_num].cpu(), size=(H, W), mode="bicubic"
                )
                refer_temporal_image = torch.cat(
                    [
                        refer_temporal_image.to(self.device),
                        torch.zeros(
                            3,
                            clip_len - refer_t_num,
                            height,
                            width,
                            device=self.device,
                            dtype=self.torch_dtype,
                        ),
                    ],
                    dim=1,
                )
            else:
                refer_temporal_image = (
                    torch.zeros(3, clip_len, height, width).to(self.device).to(self.vae.dtype)
                )
            ref_t_latents = self.vae.encode([refer_temporal_image.to(self.device)])
            msk_ref_t = self.get_i2v_mask(lat_t, lat_h, lat_w, refer_t_num, device=self.device)

            if (
                training
                and (refer_temporal_image is not None and refer_t_num > 0)
                and (not enhancing)
            ):
                ref_t_latents[0] = apply_disturbance(
                    ref_t_latents[0], refer_t_num=refer_t_num, noise_pro=0.3
                )

            y_ref_t = torch.concat([msk_ref_t, ref_t_latents[0]]).to(
                dtype=self.torch_dtype, device=self.device
            )
            y = torch.concat([y_ref, y_ref_t], dim=1)
            inputs_shared["y"] = [y]

            self.load_models_to_device(["clip"])
            img = refer_image[0, :]
            clip_context = self.clip.visual([img[:, None, :, :]]).to(
                dtype=torch.bfloat16, device=self.device
            )
            inputs_shared["clip_fea"] = clip_context

            extra_refer_num = inputs_shared.get("extra_refer_num", 0)
            target_shape = [lat_t + 1 + extra_refer_num, lat_h, lat_w]
            noise = [
                torch.randn(
                    16,
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g,
                )
            ]

            inputs_shared["noise"] = noise
            inputs_shared["latents"] = noise

        max_seq_len = int(math.ceil(np.prod(target_shape) // 4 / self.sp_size)) * self.sp_size
        if max_seq_len % self.sp_size != 0:
            raise ValueError(
                f"max_seq_len {max_seq_len} is not divisible by sp_size {self.sp_size}"
            )
        inputs_shared["seq_len"] = max_seq_len

        return inputs_posi, inputs_nega, inputs_shared

    def process_mesh_rendering(self, inputs_shared):
        def mesh_rendering(info, image_height, image_width, add_kpt_points=False):
            ehm_deform_res = self.ehm(info["smplx_params"], info["flame_params"], info["id_params"])
            focal_length_batch = torch.cat(
                [info["smplx_params"]["focal_px"], info["smplx_params"]["focal_py"]], dim=-1
            )
            batch_size = 1
            princpt_batch = (
                torch.tensor([image_width / 2, image_height / 2], dtype=torch.float32)
                .to(self.device)
                .repeat(batch_size, 1)
            )
            render_cam_param = {
                "w2c_RT": info["smplx_params"]["w2c_RT"],
                "image_size": (image_width, image_height),
                "focal_length": focal_length_batch,
                "princpt": princpt_batch,
            }
            mesh_image = self.mesh_renderer.render_mesh(
                ehm_deform_res["vertices"],
                render_cam_param,
                verts_features=self.vertices_color,
                with_wall=True,
            )
            if add_kpt_points:
                landmarks_dict = self.ehm.get_landmarks(**ehm_deform_res)
                landmarks_mp = landmarks_dict["landmarks_mp"]
                landmarks_whole = landmarks_dict["landmarks_whole"]
                landmarks_irs = ehm_deform_res["vertices"][:, self.ehm.smplx.smplx2flame_ind, :][
                    :, self.ehm.eye_mp_idx, :
                ]
                landmarks_mp_img, _ = perspective_projection(
                    info["smplx_params"]["w2c_RT"],
                    info["smplx_params"],
                    landmarks_mp,
                    image_height,
                    image_width,
                )
                landmarks_mp_img = landmarks_mp_img[0, :, :2].detach().cpu().numpy()
                landmarks_whole_img, _ = perspective_projection(
                    info["smplx_params"]["w2c_RT"],
                    info["smplx_params"],
                    landmarks_whole,
                    image_height,
                    image_width,
                )
                landmarks_whole_img = landmarks_whole_img[0, :, :2].detach().cpu().numpy()
                landmarks_irs_img, _ = perspective_projection(
                    info["smplx_params"]["w2c_RT"],
                    info["smplx_params"],
                    landmarks_irs,
                    image_height,
                    image_width,
                )
                landmarks_irs_img = landmarks_irs_img[0, :, :2].detach().cpu().numpy()
                mesh_image = draw_135_MP_keypoints(
                    landmarks_whole_img,
                    landmarks_mp_img,
                    landmarks_irs_img,
                    self.ehm.flame.lmk_mp_indices,
                    mesh_image,
                )

            return mesh_image

        self.load_models_to_device(["ehm"])
        self.ehm = self.ehm.to(device=self.device)
        add_kpt_points = True
        with torch.cuda.amp.autocast(enabled=False):
            self.ehm = self.ehm.float()
            source_info_32 = to_float32_recursive(inputs_shared["source_info"])
            image_height, image_width = (
                source_info_32["image"].shape[-2],
                source_info_32["image"].shape[-1],
            )

            source_mesh_image = mesh_rendering(
                source_info_32, image_height, image_width, add_kpt_points=add_kpt_points
            )
            box = inputs_shared["source_info"]["head_box"][0]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            source_mesh_head_image = source_mesh_image[y1:y2, x1:x2, :]
            source_mesh_head_image = (
                resize_and_pad_images(torch.tensor(source_mesh_head_image)).cpu().numpy()
            )
            rendering_images = []
            for target_info in inputs_shared["target_infos"]:
                target_info_32 = to_float32_recursive(target_info)
                mesh_image = mesh_rendering(
                    target_info_32, image_height, image_width, add_kpt_points=add_kpt_points
                )
                rendering_images.append(mesh_image)

        rendering_video = self.preprocess_video(rendering_images)

        inputs_shared["rendering_video"] = rendering_video.to(
            device=self.device, dtype=self.torch_dtype
        )
        inputs_shared["source_rendering_image"] = torch.tensor(
            source_mesh_image.astype(np.float32),
            device=self.device,
        )
        return inputs_shared

    def process_rendering_vae(self, inputs_shared):
        self.load_models_to_device(["vae"])
        source_rendering_image = inputs_shared["source_rendering_image"]
        source_rendering_image = self.preprocess_image(source_rendering_image).transpose(0, 1)
        source_rendering_latents = self.vae.encode(
            [source_rendering_image.to(self.device).to(self.torch_dtype)]
        )
        inputs_shared["source_rendering_latents"] = source_rendering_latents

        rendering_video = inputs_shared["rendering_video"]
        rendering_latents = self.vae.encode(rendering_video.to(self.device).to(self.torch_dtype))
        rendering_latents[0] = rendering_latents[0].to(dtype=self.torch_dtype, device=self.device)
        inputs_shared["rendering_latents"] = rendering_latents
        return inputs_shared

    def get_scheduler(self, sample_solver, shift=5.0, sampling_steps=20):
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=self.device, sigmas=sampling_sigmas
            )
        elif sample_solver == "flowmatch":
            sample_scheduler = FlowMatchScheduler(
                shift=shift,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=self.num_train_timesteps,
            )
            sample_scheduler.set_timesteps(sampling_steps)
            timesteps = sample_scheduler.timesteps
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler, timesteps

    @torch.no_grad()
    def __call__(
        self,
        data=None,
        clip_len=81,
        refer_t_num=5,
        cfg_scale: Optional[float] = 1.0,
        num_inference_steps: Optional[int] = 20,
        prompt: Optional[str] = "",
        negative_prompt: Optional[str] = "",
        sample_solver="flowmatch",
        input_image: Optional[Image.Image] = None,
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        add_refer=False,
        seed=-1,
        color_correction_strength=0.0,
        progress_bar_cmd=tqdm,
    ):
        extra_outputs = {}
        assert refer_t_num == 1 or refer_t_num == 5, "refer_t_num should be 1 or 5."
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        if negative_prompt == "":
            negative_prompt = self.sample_neg_prompt

        if prompt == "":
            prompt = self.sample_prompt
        if data is not None:
            real_frame_len = len(data["target_infos"])
            target_len = self.get_valid_len(real_frame_len, clip_len, overlap=refer_t_num)
            data["target_infos"] = inputs_padding(data["target_infos"], target_len)
            data["target_infos"] = to_device_recursive(data["target_infos"], device=self.device)
            data["source_info"] = to_device_recursive(data["source_info"], device=self.device)
        elif animate_pose_video is not None and animate_face_video is not None:
            real_frame_len = len(animate_pose_video)
            target_len = self.get_valid_len(real_frame_len, clip_len, overlap=refer_t_num)
            animate_pose_video = inputs_padding(animate_pose_video, target_len)
            animate_face_video = inputs_padding(animate_face_video, target_len)
        else:
            raise ValueError(
                "Either 'data' or both 'animate_pose_video' and 'animate_face_video' must be provided."
            )

        print("real frames: {} target frames: {}".format(real_frame_len, target_len))
        start = 0
        end = clip_len
        all_out_frames = []
        all_rendering_frames = []
        inputs_posi, inputs_nega = {}, {}
        refer_temporal_image = None
        refer_new_latent, refer_new_rendering_latent, refer_new_image = None, None, None

        while True:
            if start + refer_t_num >= target_len:
                break
            if start > 0:
                refer_temporal_image = all_out_frames[-1][0, :, -refer_t_num:].clone().detach()
            sample_scheduler, timesteps = self.get_scheduler(
                sample_solver, shift=5.0, sampling_steps=num_inference_steps
            )
            inputs_posi.update(
                {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                }
            )
            inputs_nega.update(
                {
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                }
            )
            if input_image is not None:
                data = {
                    "source_info": None,
                    "target_infos": None,
                    "source_image": input_image,
                    "head_image": None,
                }
            inputs_shared = {
                "source_image": data["source_image"],
                "head_image": data["head_image"],
                "source_info": data["source_info"],
                "target_infos": data["target_infos"][start:end]
                if data["target_infos"] is not None
                else None,
                "seed_g": seed_g,
                "cfg_scale": cfg_scale,
                "clip_len": clip_len,
                "refer_t_num": refer_t_num if start > 0 else 0,
                "refer_temporal_image": refer_temporal_image,
                "animate_pose_video": animate_pose_video[start:end]
                if animate_pose_video is not None
                else None,
                "animate_face_video": animate_face_video[start:end]
                if animate_face_video is not None
                else None,
                "add_refer": add_refer,
                "refer_new_latent": refer_new_latent,
                "refer_new_rendering_latent": refer_new_rendering_latent,
                "refer_new_image": refer_new_image,
            }
            if input_image is not None:
                raise ValueError("Use tracked EHM inputs for generation.")
            inputs_posi, inputs_nega, inputs_shared = self.process(
                inputs_posi, inputs_nega, inputs_shared
            )

            self.load_models_to_device(self.in_iteration_models)
            models = {name: getattr(self, name) for name in self.in_iteration_models}
            for i, t in enumerate(tqdm(timesteps)):
                latents = inputs_shared["latents"]
                timestep = [t]
                timestep = torch.stack(timestep)

                with (
                    torch.autocast(
                        device_type=torch.device(self.device).type,
                        dtype=self.torch_dtype,
                        enabled=True,
                    ),
                ):
                    noise_pred_cond = TensorList(
                        self.model_fn(
                            **models,
                            **inputs_shared,
                            **inputs_posi,
                            x=TensorList(latents),
                            t=timestep,
                        )
                    )
                    if cfg_scale > 1.0:
                        noise_pred_uncond = TensorList(
                            self.model_fn(
                                **models,
                                **inputs_shared,
                                **inputs_nega,
                                x=TensorList(latents),
                                t=timestep,
                            )
                        )
                        noise_pred = noise_pred_uncond + cfg_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                    else:
                        noise_pred = noise_pred_cond

                temp_x0 = sample_scheduler.step(
                    noise_pred[0].unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latents[0] = temp_x0.squeeze(0)
                x0 = latents
                inputs_shared["latents"] = latents
            extra_refer_num = inputs_shared.get("extra_refer_num", 0)
            self.load_models_to_device(["vae"])
            x0 = [x.to(dtype=torch.float32) for x in x0]
            out_frames = torch.stack(self.vae.decode([x0[0][:, 1 + extra_refer_num :]]))
            if color_correction_strength > 0.0:
                refer_color_image = (
                    refer_new_image[None, :, None]
                    if refer_new_image is not None
                    else out_frames[:, :, 0:1]
                )
                out_frames = match_and_blend_colors(
                    out_frames, refer_color_image, color_correction_strength
                )

            cur_refer_t_num = refer_t_num if start > 0 else 0
            out_frames = out_frames[:, :, cur_refer_t_num:]

            all_out_frames.append(out_frames.cpu())
            if "rendering_video" in inputs_shared:
                all_rendering_frames.append(
                    inputs_shared["rendering_video"][:, :, cur_refer_t_num:].cpu()
                )
            start += clip_len - refer_t_num
            end += clip_len - refer_t_num

            if cur_refer_t_num == 0 and add_refer:
                refer_new_latent = x0[0][None, :, 1:2].clone()
                refer_new_image = out_frames[0, :, 0]
                refer_new_rendering_latent = inputs_shared["rendering_latents"][0][:, 0:1]
                inputs_shared["refer_new_latent"], inputs_shared["refer_new_rendering_latent"] = (
                    refer_new_latent,
                    refer_new_rendering_latent,
                )
                inputs_shared["refer_new_image"] = refer_new_image

        videos = torch.cat(all_out_frames, dim=2)[:, :, :real_frame_len]
        extra_outputs["video_tensor"] = videos
        if len(all_rendering_frames) > 0:
            rendering_videos = torch.cat(all_rendering_frames, dim=2)[:, :, :real_frame_len]
            extra_outputs["rendering_video_tensor"] = rendering_videos
        video = self.vae_output_to_video(videos)
        self.load_models_to_device([])

        return video, extra_outputs


def model_fn_wan_video(
    dit: WanAnimateModel,
    extra_animate_adapter: Extra_Animate_Adapter = None,
    x: torch.Tensor = None,
    t: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_fea: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    seq_len: Optional[int] = None,
    pose_latents=None,
    face_pixel_values=None,
    rendering_latents=None,
    refer_image_latents=None,
    refer_head_latents=None,
    source_rendering_latents=None,
    source_rendering_head_latents=None,
    extra_refer_num=0,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    device = x.device
    if dit.freqs.device != device:
        dit.freqs = dit.freqs.to(device)
    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
    x = [dit.patch_embedding(u.unsqueeze(0)) for u in x]

    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = dit.after_patch_embedding(x, pose_latents, face_pixel_values)
    if rendering_latents is not None:
        x = extra_animate_adapter.rendering_inject(x, rendering_latents, extra_refer_num)

    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat(
        [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]
    )
    with amp.autocast(dtype=torch.float32):
        e = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t.to(device)).float())
        e0 = dit.time_projection(e).unflatten(1, (6, dit.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = dit.text_embedding(
        torch.stack(
            [torch.cat([u, u.new_zeros(dit.text_len - u.size(0), u.size(1))]) for u in context]
        )
    )

    if dit.use_img_emb:
        context_clip = dit.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=dit.freqs,
        context=context,
        context_lens=context_lens,
    )
    if dit.use_context_parallel:
        x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    for idx, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )

        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x,
                **kwargs,
                use_reentrant=False,
            )
        else:
            x = block(x, **kwargs)

        if pose_latents is not None and face_pixel_values is not None:
            x = dit.after_transformer_block(idx, x, motion_vec)
    x = dit.head(x, e)

    if dit.use_context_parallel:
        x = gather_forward(x, dim=1)

    x = dit.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def create_custom_forward(module):
    def custom_forward(*args, **kwargs):
        return module(*args, **kwargs)

    return custom_forward
