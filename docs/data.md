# Prepare training data

EMOSH learns from RGB video frames paired with EHM tracking parameters. Track each video once, then pack the results in the EMOSH environment. The original fixed training prompt is used automatically.

## 1. Track your videos

Install [EMOSH-Tracker](https://github.com/EastbeanZhang/EMOSH-Tracker) and prepare its model files. In the Tracker environment, run:

```bash
python track.py --input /path/to/video.mp4 --output results \
  --visualize --save-images
```

The default keeps the video's original dimensions. For portraits or cropped upper bodies, add `--head-only`. The optional `--target-size WIDTH HEIGHT` selects a fitting canvas while preserving the input aspect ratio.

The result directory is named after the input video:

```text
results/video/
├── input/                 # Fitted RGB frames
├── optim_tracking.npy     # Pose, expression and cameras
├── id_share_params.npy    # Shared identity and shape
├── metadata.json          # Dimensions, FPS and frame indices
└── tracking.mp4           # Input, mesh and overlay preview
```

Inspect the preview before using a record. Keep the input PNGs and their matching parameters together; independently resizing the PNGs would change their alignment with the fitted camera.

## 2. Pack training records

Activate the EMOSH environment and run from the EMOSH repository:

```bash
python prepare_dataset.py --tracking /path/to/tracker/results/video \
  --output data/train --id clip_0001
python prepare_dataset.py --tracking /path/to/another/tracker/result \
  --output data/train --id clip_0002 --split valid
```

Use unique record IDs containing letters, numbers, underscores or hyphens. The converter checks frame completeness and consistent RGB dimensions before creating a record. It preserves frame order, stores paths relative to the dataset root and retains the recorded FPS and source frame indices. Only the required arrays, frames and selected metadata are packed; the Tracker's local input path is excluded.

```text
data/train/
├── videos_info.json
├── videos_split.json
├── clip_0001/
│   ├── image_lmdb/
│   ├── optim_tracking.npy
│   ├── id_share_params.npy
│   └── prompt.txt
└── clip_0002/
    └── ...
```

`videos_info.json` maps IDs to relative record directories, ordered frame keys, frame counts and dimensions. `videos_split.json` assigns records to `train` or `valid`. A held-out record enables the short- and long-clip validation used during training.

## 3. Train

```bash
python train.py --dataset data/train --model-path models/wan-base \
  --assets assets --output outputs/train --checkpoint-offload
```

Training samples a reference frame and a consecutive target clip from the same record. The default clip length is 77 frames. Clips use 4n+1 frames; shorter records are trimmed to that length. Keep at least nine frames per record when using five-frame temporal conditioning.

## Image dimensions

The training and inference CLIs support the same resolution options:

| Mode | Arguments | Behavior |
| --- | --- | --- |
| Original dimensions | Omit resolution arguments | Pad each record to multiples of 16 without stretching or cropping its content. |
| Wan-Animate area budget | `--resolution-area 1280 720` | Preserve aspect ratio within a 1280 × 720 pixel budget and align the canvas to multiples of 16. |
| Custom canvas | `--width WIDTH --height HEIGHT` | Fit the complete image into the chosen canvas, preserving aspect ratio. Both dimensions must be multiples of 16 and at least 64. |

Wan-Animate's [official preprocessing default](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/preprocess_data.py) is an area budget, rather than a fixed orientation. The same budget supports landscape and portrait inputs.

Every image transform also updates camera intrinsics, focal lengths and full-image crop matrices. Head and hand crops remain aligned through their original-image transforms. Pose, identity and world-space geometry are unchanged. The same tracked record can therefore be used at another training canvas without fitting it again.

Training uses one clip per process, so records may have different dimensions. For inference, the reference record determines the shared output canvas; driving records are fitted inside the reference image area with their corresponding camera transforms. In original-dimension mode, export removes the temporary alignment padding and restores the reference dimensions, including odd-sized videos. Playback uses the driving record's FPS when available; `--fps` overrides it.

See [training and inference](training.md) for checkpoints, distributed training and animation.
