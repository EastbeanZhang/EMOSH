# Training and inference

Prepare the [environment](installation.md), [model files](models.md) and [tracked data](data.md) first.

## Train

```bash
python train.py --dataset data/train --model-path models/wan-base \
  --assets assets --output outputs/train --num-frames 77 \
  --height 512 --width 512 --epochs 25 --lora-rank 32 \
  --learning-rate 1e-4 --checkpoint-offload
```

The public Wan initialization remains frozen except for DiT LoRA projections. The mesh-control embedding is trained jointly with a lower learning rate of `1e-5`. Training uses a flow-matching objective, random reference augmentation and optional five-frame temporal conditioning. The default temporal-conditioning probability is 0.5. `--checkpoint-offload` moves saved activations to CPU during backward recomputation.

Set `--max-steps` to limit optimizer updates for a short run. `--gradient-accumulation N` accumulates gradients before updating the parameters. Non-finite losses or gradients stop training with an error.

For multiple local GPUs:

```bash
accelerate launch --config_file configs/ddp.yaml --num_processes 2 train.py \
  --dataset data/train --model-path models/wan-base --assets assets \
  --output outputs/train --num-frames 77 --checkpoint-offload
```

Each process uses one video clip. The effective global batch size is the process count multiplied by `--gradient-accumulation`. This recipe uses distributed data parallelism; each GPU holds a model replica.

## Save and resume

Each checkpoint directory contains:

- `model.safetensors`: trainable LoRA and mesh-control parameters, with their original parameter names.
- `training.pt`: optimizer state, optimizer step and data-loop position.
- `progress.json`: the saved step number.

For example, save every 100 optimizer updates and resume from the saved directory:

```bash
python train.py --dataset data/train --model-path models/wan-base \
  --output outputs/train --save-every 100 --checkpoint-offload
python train.py --dataset data/train --model-path models/wan-base \
  --output outputs/train --resume outputs/train/step-100 --checkpoint-offload
```

Use the same model, LoRA rank and data configuration when resuming. The optimizer and loop position are restored; the resumed run starts a new random sampling stream. `training.jsonl` records losses at optimizer updates.

## Animate

Prepare reference and driving records with `prepare_dataset.py`, then load a checkpoint produced by training:

```bash
python infer.py --dataset data/train --reference clip_0001 --driving clip_0002 \
  --model-path models/wan-base --assets assets \
  --checkpoint outputs/train/step-100/model.safetensors \
  --output outputs/animation.mp4 --num-frames 77 --steps 20 --fps 30
```

The reference record provides the image and identity/shape parameters; the driving record provides pose, expression and camera parameters. The code retargets the driving motion to the reference identity before rendering the control maps.

Long videos use overlapping chunks with five temporal reference frames. Spatially-aligned conditioning is enabled by default; `--disable-spatial-anchor` disables the additional first-chunk anchor. Match `--fps` to the intended playback rate.

For a small CUDA graph check that requires no model downloads:

```bash
python -m scripts.smoke_test --output outputs/smoke
```

This check uses a reduced-width backbone and verifies backward propagation, optimizer updates and checkpoint restoration. It does not measure the quality or memory requirements of the full model. Full-model validation is documented separately in [testing.md](testing.md).
