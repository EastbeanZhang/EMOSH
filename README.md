<div align="center">

# EMOSH: Expressive Motion and Shape Disentanglement for Human Animation

**Dongbin Zhang**<sup>1,2\*</sup>, **Hao Liu**<sup>2</sup>, **Binquan Dai**<sup>1</sup>, **Kangjie Chen**<sup>1</sup>, **Chuming Wang**<sup>1</sup>, **Chen Li**<sup>2†</sup>, **Jing LYU**<sup>2</sup>, **Haoqian Wang**<sup>1†</sup>

<sup>1</sup> Tsinghua University &nbsp; | &nbsp; <sup>2</sup> WeChat Vision, Tencent Inc.

<sup>\*</sup> Intern &nbsp; | &nbsp; <sup>†</sup> Corresponding Authors

## **ECCV 2026**

[![Project Page](https://img.shields.io/badge/Project_Page-2563EB?style=for-the-badge&logo=googlechrome&logoColor=white)](https://eastbeanzhang.github.io/EMOSH/)
[![arXiv](https://img.shields.io/badge/Paper-30343B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.28026)
[![YouTube](https://img.shields.io/badge/Demo-30343B?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=Sr9oZLtng-s)
[![Tracker](https://img.shields.io/badge/Tracker-30343B?style=for-the-badge&logo=github&logoColor=white)](https://github.com/EastbeanZhang/EMOSH-Tracker)

</div>

**We release the complete training code for EMOSH.**

EMOSH animates a reference image with expressive motion from a driving video while preserving the reference subject's body shape and identity.

<p align="center">
  <img src="assets/Docs/teaser.svg" width="900" alt="EMOSH expressive human animation and motion-shape disentanglement">
</p>

- **Expressive motion control.** Track and transfer body motion, facial expressions and hand gestures through EHM.
- **Motion–shape disentanglement.** Retarget driving motion to the reference identity before rendering hybrid mesh and keypoint controls.
- **Consistent long videos.** Combine reference, temporal and spatially-aligned conditioning across video chunks.

## Method

<p align="center">
  <img src="assets/Docs/pipeline.svg" width="900" alt="EMOSH tracking, retargeting, hybrid control and video generation pipeline">
</p>

The training implementation initializes the DiT with public **Wan2.2-Animate-14B** weights. A lightweight control embedding injects VAE-encoded EHM renderings; LoRA adapts the backbone. See the [paper](https://arxiv.org/abs/2606.28026) for the method and experiments.

## Get started

**Linux · Python 3.10 · NVIDIA CUDA GPU.** Follow the [minimal installation](docs/installation.md), then prepare the [Wan initialization and EHM assets](docs/models.md).

### Prepare tracked videos

Use [EMOSH-Tracker](https://github.com/EastbeanZhang/EMOSH-Tracker) to fit your videos and save input PNGs. Pack each result into a training record:

```bash
python prepare_dataset.py --tracking /path/to/tracker/result \
  --output data/train --id clip_0001
```

[Video → tracking → training data](docs/data.md)

### Train

```bash
python train.py --dataset data/train --model-path models/wan-base \
  --assets assets --output outputs/train --num-frames 77 \
  --lora-rank 32 --learning-rate 1e-4 --checkpoint-offload
```

Use `--max-steps` for a short run and `--resume outputs/train/step-N` to continue from a checkpoint. [Distributed training, outputs and validation](docs/training.md)

### Animate

```bash
python infer.py --dataset data/train --reference clip_0001 --driving clip_0002 \
  --model-path models/wan-base --assets assets \
  --checkpoint outputs/train/step-N/model.safetensors \
  --output outputs/animation.mp4 --num-frames 77 --steps 20
```

The reference record supplies identity and shape; the driving record supplies motion and camera. Original dimensions, Wan-Animate area budgets and custom canvases are supported.

## Motion Tracker

[EMOSH-Tracker](https://github.com/EastbeanZhang/EMOSH-Tracker) is the standalone tracker for full-body, upper-body and portrait images or videos, including flexible input resolutions. Its repository provides installation, model links and runnable examples. The [EMOSH-Tracker/](EMOSH-Tracker) submodule opens the corresponding source.

## Acknowledgements

Built on [Wan](https://github.com/Wan-Video/Wan2.2), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [EHM-Tracker](https://github.com/Pixel-Talk/EHM-Tracker), SMPL-X, FLAME and MANO. Project-authored integration is under [Apache-2.0](LICENSE); embedded components retain their [original terms](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{zhang2026emosh,
  title={EMOSH: Expressive Motion and Shape Disentanglement for Human Animation},
  author={Zhang, Dongbin and Liu, Hao and Dai, Binquan and Chen, Kangjie and Wang, Chuming and Li, Chen and LYU, Jing and Wang, Haoqian},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
