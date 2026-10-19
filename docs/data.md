# Prepare training data

Each video record contains RGB frames and the corresponding EHM tracking parameters. Prepare the videos you want to use at the training canvas size; the default recipe uses 512 × 512 and 77-frame clips.

## Track videos

Use [EMOSH-Tracker](https://github.com/EastbeanZhang/EMOSH-Tracker) in its own environment. Follow its installation and model-preparation instructions, then run:

```bash
python track.py --input /path/to/video.mp4 --output results \
  --target-size 512 512 --visualize --save-images
```

Tracker preserves the input aspect ratio on the selected canvas and exports the matching camera parameters. Keep `optim_tracking.npy`, `id_share_params.npy` and the `input/` PNGs together. The training converter reads these fitted PNGs, rather than independently resizing the original video.

## Pack records

Activate the EMOSH training environment and run from the EMOSH repository:

```bash
python prepare_dataset.py --tracking /path/to/tracker/results/video \
  --output data/train --id clip_0001
python prepare_dataset.py --tracking /path/to/another/tracker/result \
  --output data/train --id clip_0002
```

The default prompt is `A person moving.`; use `--prompt` only to provide a custom description.

The converter stores paths relative to the dataset root:

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

`videos_info.json` maps each record ID to `video_dir`, ordered `frames_keys`, `frames_number`, `width` and `height`. `videos_split.json` contains `train` and `valid` ID lists. Use `--split valid` to prepare a held-out record. Record IDs must be unique.

Training samples a reference image and a consecutive target clip from the same record. Each target clip uses 4n+1 frames. Keep at least 77 consecutive frames per record for the standard recipe. Shorter records are shortened to a valid clip length; use at least 9 frames when testing temporal conditioning.

The stored fitting canvas and the requested training dimensions must match. To change the training canvas, rerun Tracker at that size instead of resizing only the PNGs.
