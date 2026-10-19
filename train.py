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
        temporal_frames = 5 if torch.rand(()).item() < self.temporal_probability else 0
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
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    if args.num_frames < 5 or args.num_frames % 4 != 1:
        p.error("--num-frames must be 4n+1 and at least 5")
    if min(args.height, args.width) < 64 or args.height % 16 or args.width % 16:
        p.error("Height and width must be multiples of 16 and at least 64")
    if not 0 <= args.temporal_probability <= 1:
        p.error("--temporal-probability must be between 0 and 1")
    if min(args.epochs, args.save_every, args.gradient_accumulation, args.lora_rank) <= 0:
        p.error("Epochs, save interval, accumulation and LoRA rank must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        p.error("--max-steps must be positive")
    return args


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
    dataset = Tracked_VideoDataset(
        base_path=args.dataset,
        split="train",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=args.workers, collate_fn=first_sample
    )
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
    progress = {"step": 0, "epoch": 0, "batch": -1}
    if args.resume:
        progress = load_checkpoint(args.resume, model, optimizer)
    step = progress["step"]
    last_epoch, last_batch = progress["epoch"], progress["batch"]
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    output = Path(args.output)
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
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                loss_value = accelerator.gather(loss.detach().reshape(1)).mean().item()
                if accelerator.is_main_process:
                    record = {"step": step, "epoch": epoch, "loss": loss_value}
                    with (output / "training.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(f"step={step} loss={loss_value:.6f}", flush=True)
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
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
