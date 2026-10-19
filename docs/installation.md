# Installation

Use Linux, Python 3.10 and an NVIDIA CUDA GPU. The reference setup uses PyTorch 2.9.1 with CUDA 13.0 and PyTorch3D 0.7.9. A matching CUDA toolkit and C++ compiler are needed to build PyTorch3D.

```bash
git clone --recurse-submodules https://github.com/EastbeanZhang/EMOSH.git
cd EMOSH
conda create -n emosh python=3.10 -y
conda activate emosh
python -m pip install pip==25.2 setuptools==80.10.2 wheel==0.48.0
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation chumpy==0.70
python -m pip install ninja==1.13.0
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
export MAX_JOBS=4
export NVCC_FLAGS='--device-entity-has-hidden-visibility=false -static-global-template-stub=false'
python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9'
python -m pip check
```

The training path uses PyTorch scaled dot-product attention. A separate FlashAttention build, xFormers, DeepSpeed, ModelScope, Gradio, audio packages and Gaussian rasterizers are not required.

The older SMPL-X, FLAME and MANO model files use NumPy aliases retained in NumPy 1.23.5. Keep the NumPy pin when using these files.

For development checks:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -m pytest -q
```

GPU checks and their scope are recorded in [testing.md](testing.md).
