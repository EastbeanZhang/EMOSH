# Model preparation

## Public Wan initialization

Download the required files from [Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B):

```bash
python -m scripts.download_base --output models/wan-base
```

The downloader pins the model revision, selects the 16 required files (approximately 47.7 GiB), and verifies their sizes and content hashes against [base_manifest.json](../assets/base_manifest.json). It excludes the unrelated preprocessing networks and additional text-model formats.

```text
models/wan-base/
├── config.json
├── diffusion_pytorch_model.safetensors.index.json
├── diffusion_pytorch_model-00001-of-00004.safetensors
├── diffusion_pytorch_model-00002-of-00004.safetensors
├── diffusion_pytorch_model-00003-of-00004.safetensors
├── diffusion_pytorch_model-00004-of-00004.safetensors
├── Wan2.1_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
├── google/umt5-xxl/                 # tokenizer files
└── xlm-roberta-large/               # tokenizer files
```

## EHM assets

Training renders tracked motion using EHM. Prepare the same licensed SMPL-X, FLAME and MANO assets as [EMOSH-Tracker](https://github.com/EastbeanZhang/EMOSH-Tracker/blob/main/docs/models.md), retaining its directory names. The initialization networks used by Tracker are not needed when training on existing tracking results.

```text
assets/
├── smplx/    # SMPL-X 2020 neutral, correspondences and joint data
├── flame/    # FLAME 2020, eyelid offsets and landmark embeddings
└── mano/     # MANO v1.2 right hand and selected vertices
```

Use the official [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php), [FLAME](https://flame.is.tue.mpg.de/download.php) and [MANO](https://mano.is.tue.mpg.de/download.php) download pages for model registration and files. The ten auxiliary EHM files can be obtained with the Tracker helper:

```bash
python EMOSH-Tracker/scripts/download_support.py --assets assets
```

An existing asset directory can be selected with `--assets /path/to/ehm-assets` in the training and inference commands.

The same initialization is also available from the [official ModelScope repository](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B). To verify an existing directory without downloading:

```bash
python -m scripts.download_base --output models/wan-base --verify-only
```
