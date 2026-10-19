import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, inject_adapter_in_model

from models.utils.trainer_utils import load_checkpoint, save_checkpoint
from models.wan.modules.animate_adapter import Extra_Animate_Adapter
from models.wan.modules.animate_utils import TensorList
from models.wan.modules.model_animate import WanAnimateModel
from models.wan.schedulers.flow_match import FlowMatchScheduler
from models.wan.wan_video_pipe import model_fn_wan_video


def main():
    p = argparse.ArgumentParser(
        description="Check the EMOSH training graph on CUDA without model downloads."
    )
    p.add_argument("--output", default="outputs/smoke")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(2)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    dit = WanAnimateModel(
        dim=128,
        ffn_dim=256,
        freq_dim=32,
        text_dim=64,
        num_heads=8,
        num_layers=1,
        text_len=8,
        use_img_emb=False,
    )
    torch.nn.init.normal_(dit.head.head.weight, std=0.02)
    dit.requires_grad_(False)
    dit = inject_adapter_in_model(
        LoraConfig(r=4, lora_alpha=4, target_modules=["q", "k", "v", "o"]), dit
    )
    adapter = Extra_Animate_Adapter(dim=128)
    model = torch.nn.Module()
    model.pipe = torch.nn.Module()
    model.pipe.dit = dit
    model.pipe.extra_animate_adapter = adapter
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    scheduler = FlowMatchScheduler(sigma_min=0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    clean = torch.randn(1, 16, 4, 4, 4, device=device)
    noise = torch.randn_like(clean)
    timestep = scheduler.timesteps[400:401].to(device)
    inputs = TensorList([scheduler.add_noise(clean, noise, timestep)[0]])
    target = scheduler.training_target(clean, noise, timestep)
    control = [torch.randn(16, 3, 4, 4, device=device)]
    before = adapter.rendering_patch_embedding.weight.detach().clone()
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model_fn_wan_video(
                dit,
                adapter,
                x=inputs,
                t=timestep,
                context=[torch.randn(5, 64, device=device)],
                y=[torch.randn(20, 4, 4, 4, device=device)],
                rendering_latents=control,
                seq_len=16,
                use_gradient_checkpointing=True,
            )[0]
            loss = (prediction[:, 1:].float() - target[0, :, 1:]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if norm <= 0:
            raise AssertionError("No gradient reached trainable parameters")
        optimizer.step()
        losses.append(loss.item())
    if torch.equal(before, adapter.rendering_patch_embedding.weight):
        raise AssertionError("Control embedding did not update")
    output = Path(args.output)
    save_checkpoint(output / "checkpoint", model, optimizer, 2, 0, 1)
    saved = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for p in parameters:
            p.zero_()
    state = load_checkpoint(output / "checkpoint", model, optimizer)
    assert state["step"] == 2
    for name, parameter in model.named_parameters():
        if name in saved:
            torch.testing.assert_close(parameter, saved[name])
    record = {
        "scope": "reduced-width training graph",
        "losses": losses,
        "optimizer_updates": 2,
        "checkpoint_restored": True,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
