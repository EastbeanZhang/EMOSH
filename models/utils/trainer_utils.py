import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def save_checkpoint(directory, model, optimizer, step, epoch, batch):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    state = {
        name.removeprefix("pipe."): p.detach().cpu().contiguous()
        for name, p in model.named_parameters()
        if name in names
    }
    save_file(state, str(directory / "model.safetensors"))
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "batch": batch},
        directory / "training.pt",
    )
    (directory / "progress.json").write_text(json.dumps({"step": step}, indent=2) + "\n")


def load_checkpoint(directory, model, optimizer=None):
    directory = Path(directory)
    weights = load_file(str(directory / "model.safetensors"), device="cpu")
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    weights = {"pipe." + name: tensor for name, tensor in weights.items()}
    if set(weights) != expected:
        raise ValueError("Checkpoint trainable parameters do not match this configuration.")
    model.load_state_dict(weights, strict=False)
    state = torch.load(directory / "training.pt", map_location="cpu", weights_only=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


def first_sample(batch):
    return batch[0]


def finite_gradients(parameters):
    return all(torch.isfinite(p.grad).all().item() for p in parameters if p.grad is not None)
