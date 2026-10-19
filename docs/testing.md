# Runtime validation

The release is tested in a fresh Linux environment with Python 3.10.18, PyTorch 2.9.1 / CUDA 13.0, torchvision 0.24.1, PyTorch3D 0.7.9 and the pinned runtime requirements. `pip check` passes. The attention path uses native PyTorch SDPA.

## Checks

- Eight tests pass on CUDA, covering attention masks and gradients, paired reference/control augmentation, dataset conversion, identity retargeting and optimizer checkpoint restoration. Ruff passes.
- The reduced-width smoke test completes two optimizer updates and restores its checkpoint.
- Real Tracker parameters produce finite 512 × 512 hybrid mesh/keypoint control frames.
- The complete Wan2.2-Animate-14B model completes two updates on a nine-frame full-body clip. Both the mesh-control embedding and LoRA parameters change. A saved checkpoint restores the optimizer and loop position and continues through step 3.
- A 77-frame portrait clip completes a full-model update with finite loss and gradients, five-frame temporal conditioning and CPU activation offload. Both control and LoRA parameters update.
- The inference CLI loads the resulting checkpoint and exports a 13-frame, 512 × 512 video at 30 FPS, using two overlapping nine-frame chunks, two denoising steps per chunk and spatially-aligned conditioning.
- Two-GPU DDP completes one update with nine frames per GPU. Both ranks update the control and LoRA parameters, and their updated control weights have identical content hashes.

The full-model checks use BF16, LoRA rank 32, a batch size of one and gradient checkpointing. The nine-frame runs disable temporal conditioning; the longer run exercises five-frame temporal conditioning and CPU activation offload.

## GPU memory

| Run | Canvas | Frames per clip | Optimizer updates | Peak allocated | Peak reserved |
| --- | --- | --- | --- | --- | --- |
| Full-body training | 512 × 512 | 9 | 2 | 54.60 GiB | 60.58 GiB |
| Restore and continue | 512 × 512 | 9 | 1 additional | 54.59 GiB | 60.58 GiB |
| Portrait training, activation offload | 512 × 512 | 77 | 1 | 55.50 GiB | 60.90 GiB |
| Two-GPU DDP, per GPU | 512 × 512 | 9 | 1 | 54.30 GiB | 60.97 GiB |

These are PyTorch allocator peaks for the stated configurations. They exclude CUDA context memory and other processes; allow additional headroom. Shorter clips are useful for checking an installation, while the standard training recipe uses 77 frames.

Short training and low-step generation checks establish that the code runs; they do not establish convergence or reproduce the paper's generation quality.
