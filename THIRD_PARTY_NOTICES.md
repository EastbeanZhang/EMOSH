# Third-party notices

The root Apache-2.0 license applies to project-authored training and inference integration. Embedded code, model assets and media retain their original terms.

| Component | Source and terms |
| --- | --- |
| Wan backbone, encoders, model helpers and schedulers | [Wan2.2](https://github.com/Wan-Video/Wan2.2), Apache-2.0; existing headers retained |
| Training/pipeline and flow-matching utilities adapted in the research code | [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), Apache-2.0 |
| SMPL-X, FLAME, MANO and EHM decoder | [SMPL-X](https://smpl-x.is.tue.mpg.de/), [FLAME](https://flame.is.tue.mpg.de/), [MANO](https://mano.is.tue.mpg.de/) and [EHM-Tracker](https://github.com/Pixel-Talk/EHM-Tracker); [SMPL-X source license](licenses/SMPLX-LICENSE.txt), embedded notices and separately registered model agreements apply |
| CLIP and OpenCLIP ancestry | [CLIP MIT license](licenses/CLIP-MIT.txt) and [OpenCLIP MIT license](licenses/OpenCLIP-MIT.txt); source credits retained in the Wan headers |
| Motion encoder adapted from LIA | [LIA](https://github.com/wyhsirius/LIA), by Yaohui Wang, Di Yang, François Brémond and Antitza Dantcheva; [CC BY-NC 4.0](licenses/LIA-LICENSE.md). The adapted code in `models/wan/modules/motion_encoder.py` retains its source credit and noncommercial terms. |
| LMDB reader ancestry | Xuangeng Chu attribution retained in `models/utils/lmdb.py` |
| PyTorch3D, Transformers, Diffusers, PEFT and installed libraries | Their respective upstream licenses |

The vendored source has been adapted: experiment branches and unused interfaces were removed, attention uses PyTorch SDPA, asset locations are configurable, and training/export entry points are consolidated. It is not an unmodified upstream snapshot.

The paper teaser and pipeline figures retain the paper authors' rights. Public initialization weights and separately downloaded EHM assets retain the terms of their original distributions.
