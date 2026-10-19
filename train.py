import argparse
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from peft import LoraConfig, inject_adapter_in_model
from tqdm import tqdm

from models.utils.data_utils import Tracked_VideoDataset
from models.utils.general_utils import save_video
from models.utils.image_geometry import add_resolution_arguments, validate_resolution_arguments
from models.utils.trainer_utils import (
    finite_gradients,
    first_sample,
    load_checkpoint,
    save_checkpoint,
)
from models.wan.configs import WAN_CONFIGS
from models.wan.wan_video_pipe import WanVideoPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


class EMOSHTrainingModule(torch.nn.Module):
    def __init__(
        self, model_path, assets, lora_rank=32, temporal_probability=0.5, checkpoint_offload=False
    ):
        super().__init__()
        self.pipe = WanVideoPipeline.from_pretrained(
            config=WAN_CONFIGS["animate-14B"],
            checkpoint_dir=model_path,
            torch_dtype=torch.bfloat16,
            device="cpu",
            offload_device="cpu",
            convert_model_dtype=True,
            use_relighting_lora=False,
            assets=assets,
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([])
        self.pipe.dit = inject_adapter_in_model(
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
            ),
            self.pipe.dit,
        )
        self.pipe.extra_animate_adapter.requires_grad_(True)
        self.temporal_probability = temporal_probability
        self.checkpoint_offload = checkpoint_offload

    def to(self, *args, **kwargs):
        self.pipe.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        super().train(mode)
        for name in ["text_encoder", "clip", "vae", "ehm"]:
            getattr(self.pipe, name).eval()
        return self

    def forward(self, data):
        temporal_frames = 0
        if self.temporal_probability > 0 and torch.rand(1).item() < self.temporal_probability:
            temporal_frames = 5
        inputs = {
            "height": data["target_video"][0].shape[0],
            "width": data["target_video"][0].shape[1],
            "clip_len": len(data["target_video"]),
            "seed_g": None,
            "source_image": data["source_image"],
            "head_image": data["head_image"],
            "source_info": data["source_info"],
            "target_infos": data["target_infos"],
            "input_video": data["target_video"],
            "cfg_scale": 1,
            "refer_t_num": temporal_frames,
            "use_gradient_checkpointing": True,
            "use_gradient_checkpointing_offload": self.checkpoint_offload,
        }
        positive, _, inputs = self.pipe.process(
            {"prompt": data["prompt"]}, {}, inputs, training=True
        )
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        return self.pipe.training_loss(**models, **inputs, **positive)


def parse_args():
    p = argparse.ArgumentParser(description="Train EMOSH with tracked human videos.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--assets", default="assets")
    p.add_argument("--output", default="outputs/train")
    add_resolution_arguments(p)
    p.add_argument("--num-frames", type=int, default=77)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--temporal-probability", type=float, default=0.5)
    p.add_argument("--checkpoint-offload", action="store_true")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cache-records", type=int, default=8)
    p.add_argument("--evaluate-every", type=int, default=500)
    p.add_argument("--validation-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    if args.num_frames < 5 or args.num_frames % 4 != 1:
        p.error("--num-frames must be 4n+1 and at least 5")
    validate_resolution_arguments(p, args)
    if args.num_frames < 9 and args.temporal_probability > 0:
        p.error("Five-frame temporal conditioning requires at least nine video frames")
    if not 0 <= args.temporal_probability <= 1:
        p.error("--temporal-probability must be between 0 and 1")
    if min(args.epochs, args.save_every, args.gradient_accumulation, args.lora_rank) <= 0:
        p.error("Epochs, save interval, accumulation and LoRA rank must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        p.error("--max-steps must be positive")
    if args.evaluate_every < 0 or args.validation_steps <= 0:
        p.error("Evaluation interval must be nonnegative and validation steps positive")
    if args.workers < 0 or args.cache_records < 1:
        p.error("Worker count must be nonnegative and cache capacity positive")
    return args


@torch.no_grad()
def validate(model, loaders, accelerator, output, step, args):
    pipe = accelerator.unwrap_model(model).pipe
    losses = []
    directory = output / "validation" / f"step-{step}"
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    try:
        for name, loader in loaders:
            for index, data in enumerate(loader):
                _, result = pipe(
                    data=data,
                    prompt=data["prompt"],
                    clip_len=args.num_frames,
                    num_inference_steps=args.validation_steps,
                    cfg_scale=1.0,
                    seed=0,
                )
                video = result["video_tensor"]
                target = pipe.preprocess_video(data["target_video"], ori_range=1.0).to(video.device)
                loss = (video.float() - target.float()).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite validation MSE")
                losses.extend(
                    accelerator.gather(loss.to(accelerator.device).reshape(1)).cpu().tolist()
                )
                panels = [video]
                if "rendering_video_tensor" in result:
                    panels.append(result["rendering_video_tensor"].to(video.device))
                panels.append(target)
                frames = pipe.vae_output_to_video(torch.cat(panels, dim=-1))
                save_video(
                    frames,
                    directory / f"{name}-{accelerator.process_index}-{index}.mp4",
                    fps=15,
                    quality=4,
                )
        if not losses:
            raise ValueError("Validation needs at least one held-out record.")
        return sum(losses) / len(losses)
    finally:
        model.train()


def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError("Training requires an NVIDIA CUDA GPU.")
    set_seed(args.seed, device_specific=True)
    if args.workers:
        # Tracked clips contain many small tensors; avoid one open descriptor per tensor.
        torch.multiprocessing.set_sharing_strategy("file_system")
    dataset = Tracked_VideoDataset(
        base_path=args.dataset,
        split="train",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        resolution_area=args.resolution_area,
        cache_size=args.cache_records,
    )
    minimum_frames = 9 if args.temporal_probability > 0 else 5
    if dataset.minimum_frames < minimum_frames:
        raise ValueError(
            f"Training records need at least {minimum_frames} frames for this configuration."
        )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=args.workers, collate_fn=first_sample
    )
    validation_loaders = []
    if args.evaluate_every and dataset.videos_info.summary("valid")[0]:
        if args.num_frames < 9:
            raise ValueError(
                "Chunked validation needs at least nine frames; use --evaluate-every 0 to disable it."
            )
        for name, frames in [("short", args.num_frames), ("long", 2 * args.num_frames - 5)]:
            held_out = Tracked_VideoDataset(
                base_path=args.dataset,
                split="valid",
                height=args.height,
                width=args.width,
                num_frames=frames,
                resolution_area=args.resolution_area,
                cache_size=args.cache_records,
            )
            held_out.slice(
                min(accelerator.num_processes, len(held_out)),
                start=accelerator.num_processes if name == "short" else 0,
            )
            val_loader = torch.utils.data.DataLoader(
                held_out,
                batch_size=1,
                num_workers=args.workers,
                collate_fn=first_sample,
            )
            validation_loaders.append((name, accelerator.prepare(val_loader)))
    model = EMOSHTrainingModule(
        args.model_path,
        args.assets,
        args.lora_rank,
        args.temporal_probability,
        args.checkpoint_offload,
    )
    adapter = model.pipe.extra_animate_adapter.rendering_patch_embedding
    special = set(map(id, adapter.parameters()))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(adapter.parameters()), "lr": 1e-5},
            {"params": [p for p in parameters if id(p) not in special], "lr": args.learning_rate},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    progress = {"step": 0, "epoch": 0, "batch": -1}
    if args.resume:
        progress = load_checkpoint(args.resume, model, optimizer, scheduler)
    step = progress["step"]
    last_epoch, last_batch = progress["epoch"], progress["batch"]
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    model.train()
    output = Path(args.output)
    best_validation_loss = float("inf")
    validation_log = output / "validation.jsonl"
    if args.resume and validation_log.is_file():
        prior = [json.loads(line) for line in validation_log.read_text().splitlines()]
        best_validation_loss = min(
            (item["mse"] for item in prior if item["step"] > 100), default=float("inf")
        )
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        print(f"Trainable parameters: {sum(p.numel() for p in parameters):,}")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(progress["epoch"], args.epochs):
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        for batch, data in enumerate(tqdm(loader, disable=not accelerator.is_main_process)):
            if epoch == progress["epoch"] and batch <= progress["batch"]:
                continue
            if args.max_steps is not None and step >= args.max_steps:
                break
            with accelerator.accumulate(model):
                loss = model(data)
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not finite_gradients(parameters):
                        raise FloatingPointError(f"Non-finite gradient at step {step}")
                    accelerator.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                loss_value = accelerator.gather(loss.detach().reshape(1)).mean().item()
                if accelerator.is_main_process:
                    record = {"step": step, "epoch": epoch, "loss": loss_value}
                    with (output / "training.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(f"step={step} loss={loss_value:.6f}", flush=True)
                if validation_loaders and step % args.evaluate_every == 0:
                    validation_loss = validate(
                        model, validation_loaders, accelerator, output, step, args
                    )
                    if accelerator.is_main_process:
                        with (output / "validation.jsonl").open("a") as stream:
                            stream.write(json.dumps({"step": step, "mse": validation_loss}) + "\n")
                        if step > 100 and validation_loss < best_validation_loss:
                            save_checkpoint(
                                output / "best",
                                accelerator.unwrap_model(model),
                                optimizer,
                                step,
                                epoch,
                                batch,
                                scheduler,
                            )
                    if step > 100:
                        best_validation_loss = min(best_validation_loss, validation_loss)
                        if accelerator.is_main_process:
                            save_checkpoint(
                                output / "latest",
                                accelerator.unwrap_model(model),
                                optimizer,
                                step,
                                epoch,
                                batch,
                                scheduler,
                            )
                if step % args.save_every == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_checkpoint(
                            output / f"step-{step}",
                            accelerator.unwrap_model(model),
                            optimizer,
                            step,
                            epoch,
                            batch,
                            scheduler,
                        )
            last_epoch, last_batch = epoch, batch
        if args.max_steps is not None and step >= args.max_steps:
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            output / f"step-{step}",
            accelerator.unwrap_model(model),
            optimizer,
            step,
            last_epoch,
            last_batch,
            scheduler,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
