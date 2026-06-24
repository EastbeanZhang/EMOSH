# EMOSH: Expressive Motion and Shape Disentanglement for Human Animation

**Dongbin Zhang**<sup>1,2\*</sup>, **Hao Liu**<sup>2</sup>, **Binquan Dai**<sup>1</sup>, **Kangjie Chen**<sup>1</sup>, **Chuming Wang**<sup>1</sup>, **Chen Li**<sup>2†</sup>, **Jing LYU**<sup>2</sup>, **Haoqian Wang**<sup>1†</sup>

<sup>1</sup> Tsinghua University &nbsp; | &nbsp; <sup>2</sup> WeChat Vision, Tencent Inc.

<sup>\*</sup> Intern &nbsp; | &nbsp; <sup>†</sup> Corresponding Authors

**ECCV 2026**

---

[![Project Page](https://img.shields.io/badge/Project-Page-EC4899?style=for-the-badge)](https://eastbeanzhang.github.io/EMOSH/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-B31B1B?style=for-the-badge)](https://arxiv.org/abs/xxxx.xxxxx)
[![YouTube](https://img.shields.io/badge/YouTube-Demo-FF0000?style=for-the-badge)](https://youtu.be/xxxxxxxxxxx)

---

## Introduction

**EMOSH** is a novel framework for high-fidelity controllable human video generation. Given a reference image and a driving video, EMOSH achieves expressive, mesh-guided human animation while explicitly disentangling expressive motion from body shape — preventing shape leakage from the driving subject.

## Overview

![Teaser](emote-animate%20图/teaser/teaser.png)

Given a reference image and a driving video, EMOSH achieves high-fidelity, mesh-guided expressive human animation while disentangling expressive motion from body shape to prevent shape leakage.

## Method

![Pipeline](emote-animate%20图/pipeline/pipeline/pipeline.png)

First, the motion tracker extracts motion and camera parameters from the driving video and shape parameters from the reference image, achieving motion-shape disentanglement via EHM retargeting. The retargeted model is then rendered into hybrid control signals through semantic color shading and keypoint drawing, and encoded into motion latents. During generation, the reference latent and noisy latent are concatenated; for subsequent video chunks, the spatially-aligned and temporal latents are additionally activated. The motion latents are injected into the main sequence via addition and fed into the DiT network for denoising, and are finally decoded into the output video.

## Citation

```bibtex
@inproceedings{zhang2026emosh,
  title={EMOSH: Expressive Motion and Shape Disentanglement for Human Animation},
  author={Zhang, Dongbin and Liu, Hao and Dai, Binquan and Chen, Kangjie and Wang, Chuming and Li, Chen and LYU, Jing and Wang, Haoqian},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
