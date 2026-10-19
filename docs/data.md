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

For a collection of clips, create `clips.jsonl` with one record per line:

```jsonl
{"id":"clip_0001","tracking":"tracking/video_0001"}
{"id":"clip_0002","tracking":"tracking/video_0002"}
{"id":"clip_0003","tracking":"tracking/video_0003","split":"valid"}
```

Tracking paths are absolute or relative to the manifest file. IDs must be unique and contain letters, numbers, underscores or hyphens. Records default to `train`; reserve held-out clips with `"split":"valid"`. The original fixed prompt is supplied automatically.

Activate the EMOSH environment and run from its repository:

```bash
python prepare_dataset.py --manifest clips.jsonl \
  --output data/train --workers 8 --resume
```

The manifest is read incrementally with a bounded number of pending jobs. Workers pack separate clips; completed records are committed to a SQLite index in small transactions. This avoids rewriting a growing JSON index for every clip. Increase `--workers` according to available CPU, memory and storage bandwidth.

`--resume` skips completed records after checking the source file sizes, modification times and packing settings, along with the sizes of the packed files. It recovers interrupted writes and completed records not yet committed to the index. These checks are not content hashes: keep source files unchanged between runs. Changed sources or damaged outputs are reported rather than overwritten.

Failed clips do not discard successful clips. The command exits with status 1 if any fail and writes their IDs, manifest line numbers and errors to `packing-errors-*.jsonl` in the output directory. Fix the failed sources and rerun the same command with `--resume`. Keep the manifest and split assignments stable during retries.

For a single clip, the original command remains available:

```bash
python prepare_dataset.py --tracking /path/to/tracker/results/video \
  --output data/train --id clip_0001
python prepare_dataset.py --tracking /path/to/another/tracker/result \
  --output data/train --id clip_0002 --split valid
```

The converter checks frame completeness and consistent RGB dimensions before creating each record. It preserves frame order, relative record paths, FPS and source frame indices. PNG pixels, tracking arrays and LMDB frame keys retain their original format. Source directory paths are excluded from the packed metadata.

```text
data/train/
├── dataset.sqlite3
└── records/
    ├── <bucket>/
    │   └── clip_0001/
    │       ├── image_lmdb/
    │       ├── optim_tracking.npy
    │       ├── id_share_params.npy
    │       ├── prompt.txt
    │       └── record.json
    └── ...
```

IDs determine one of 256 directory buckets, keeping large collections out of a single flat directory. `dataset.sqlite3` stores splits and frame metadata; `record.json` supports recovery of each completed clip. Training loads the split's IDs once and reads detailed metadata on demand. LMDB, parameter and metadata caches have fixed capacities per reader. Finish packing before training and keep the dataset unchanged while it is being read. This packing entry point runs on one machine; use a local filesystem that supports file locking for its index and output directories.

### Existing datasets

Datasets with `videos_info.json` and `videos_split.json` still load directly. To create the scalable index without repacking frames or changing existing record directories:

```bash
python prepare_dataset.py --index-only --output data/train
```

When both formats exist, training and inference use SQLite. The old JSON files are not updated automatically. Export a fresh snapshot only when another tool requires it, after all packing processes have finished:

```bash
python prepare_dataset.py --index-only --output data/train --export-json
```

Legacy records without `record.json` remain readable, but cannot be skipped or overwritten by the packer's `--resume` check. Keep them in place and assign new IDs to additional clips.

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
