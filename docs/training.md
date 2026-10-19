# Training and inference

Prepare the [environment](installation.md), [model files](models.md) and [tracked data](data.md) first.

## Train

```bash
python train.py --dataset data/train --model-path models/wan-base \
  --assets assets --output outputs/train --num-frames 77 \
  --epochs 25 --lora-rank 32 \
  --learning-rate 1e-4 --checkpoint-offload
```

Resolution arguments are optional: the default keeps each record's native dimensions with temporary padding for the model. Use `--resolution-area 1280 720` for the Wan-Animate area budget, or set both `--width` and `--height` for a custom canvas. The image, camera and crop transforms remain aligned. See [image dimensions](data.md#image-dimensions).

The public Wan initialization remains frozen except for DiT LoRA projections. The mesh-control embedding is trained jointly with a lower learning rate of `1e-5`. The original `ConstantLR` schedule scales both learning rates by one third for the first five scheduler steps, then restores their configured values. Training uses a flow-matching objective, random reference augmentation and optional five-frame temporal conditioning. The default temporal-conditioning probability is 0.5. `--checkpoint-offload` moves saved activations to CPU during backward recomputation.

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
- `training.pt`: optimizer and learning-rate scheduler states, optimizer step and data-loop position.
- `progress.json`: the saved step number.

For example, save every 100 optimizer updates and resume from the saved directory:

```bash
python train.py --dataset data/train --model-path models/wan-base \
  --output outputs/train --save-every 100 --checkpoint-offload
python train.py --dataset data/train --model-path models/wan-base \
  --output outputs/train --resume outputs/train/step-100 --checkpoint-offload
```

Use the same model, LoRA rank and data configuration when resuming. The optimizer, learning-rate scheduler and loop position are restored; the resumed run starts a new random sampling stream. `training.jsonl` records losses at optimizer updates.

## Held-out validation

Prepare records with `--split valid` to enable validation. The default evaluates short and longer clips every 500 optimizer updates, using 20 denoising steps, CFG 1 and seed 0. It records pixel MSE in `validation.jsonl` and saves generated/control/target previews under `validation/`. After step 100, validation maintains `best/` and `latest/` checkpoints. With no held-out records, training proceeds without validation.

Use `--evaluate-every` to change the interval (`0` disables it) and `--validation-steps` to change the denoising steps for a runtime check.

## Animate

Prepare reference and driving records with `prepare_dataset.py`, then load a checkpoint produced by training:

```bash
python infer.py --dataset data/train --reference clip_0001 --driving clip_0002 \
  --model-path models/wan-base --assets assets \
  --checkpoint outputs/train/step-100/model.safetensors \
  --output outputs/animation.mp4 --num-frames 77 --steps 20
```

The reference record provides the image and identity/shape parameters; the driving record provides pose, expression and camera parameters. The code retargets the driving motion to the reference identity before rendering the control maps.

Long videos use overlapping chunks with five temporal reference frames. As in the original inference entry point, additional spatial anchoring is disabled by default; `--spatial-anchor` enables it. The default sampler is `flowmatch`; `--sample-solver unipc` and `--sample-solver dpm++` select the other original samplers. Export preserves frame coverage and uses the driving record's FPS, falling back to the Wan configuration's 30 FPS; `--fps` overrides it.
