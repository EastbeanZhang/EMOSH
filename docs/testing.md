# Runtime validation

Checked in a fresh Linux environment with Python 3.10.18, PyTorch 2.9.1 / CUDA 13.0, torchvision 0.24.1, PyTorch3D 0.7.9 and the pinned requirements. `pip check` and Ruff pass. The attention path uses native PyTorch SDPA.

## Coverage

- **41 regression checks pass**, covering dataset packing and loading, camera/crop transforms, odd dimensions, reference/driving canvas alignment, attention gradients, identity retargeting, temporal chunk boundaries and checkpoint restoration. The data-loading and chunk checks also pass with CUDA disabled.
- **50 full-model optimizer updates** complete on real full-body and portrait tracking records at 640 × 384, with nine frames per clip. Losses, gradients, parameters and saved optimizer states remain finite. Both temporal modes are exercised. Step 25 restores model weights, optimizer state and the original `ConstantLR` scheduler exactly, then continues to step 50.
- Held-out validation runs at steps 25 and 50, including nine-frame and thirteen-frame clips, pixel MSE and generated/control/target previews.
- A **77-frame native 1920 × 1080 portrait clip** completes a full-model update on a padded 1920 × 1088 canvas, with five-frame temporal conditioning and activation offload.
- **1280 × 720 and 720 × 1280** clips complete full-model updates and thirteen-frame exports. The original UniPC and DPM++ samplers are exercised alongside FlowMatch.
- A **459 × 639** video passes the default native-resolution Tracker CLI, dataset conversion, training on a 464 × 640 model canvas and exact 459 × 639 export at 30 FPS.
- Two-GPU DDP completes an optimizer update and held-out validation. Both ranks produce identical updated control-weight hashes.
- Inference exports one-frame, thirteen-frame and seventeen-frame videos. The seventeen-frame case spans three overlapping nine-frame chunks; native cross-resolution inputs retain the reference dimensions and the driving FPS. Spatial anchoring is checked both enabled and disabled.
- Rectangular control rendering agrees with an unbinned rasterization reference to within `3.1e-5` in pixel intensity. Rasterization capacity is sized to the mesh rather than dropping faces in crowded bins.

## Allocator peaks

All runs use the complete Wan2.2-Animate-14B model, BF16, LoRA rank 32, batch size one per GPU, gradient checkpointing and CPU activation offload. The 50-update row includes held-out validation. The 720P and 77-frame checks use five-frame temporal conditioning; the odd-sized nine-frame check disables it.

| Run | Training canvas | Frames per clip | Optimizer updates | Peak allocated | Peak reserved |
| --- | --- | --- | --- | --- | --- |
| Full-body and portrait | 640 × 384 | 9 | 50, including resume | 55.00 GiB | 59.72 GiB |
| Landscape | 1280 × 720 | 9 | 1 | 74.87 GiB | 94.61 GiB |
| Portrait | 720 × 1280 | 9 | 1 | 74.87 GiB | 94.61 GiB |
| Native odd-sized video | 464 × 640 | 9 | 1 | 55.12 GiB | 62.37 GiB |
| Native portrait, standard clip | 1920 × 1088 | 77 | 1 | 174.46 GiB | 212.29 GiB |
| Two-GPU DDP, per GPU | 640 × 384 | 9 | 1 | 55.29 GiB | 60.18 GiB |

These are measured PyTorch allocator peaks for the stated configurations, rather than minimum GPU capacities. CUDA context memory and other processes are excluded. Use an area budget or a smaller canvas and clip length to control memory.

The checks establish execution and numerical stability. They do not establish convergence or reproduce the paper's animation quality.
