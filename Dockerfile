# ------------------------------------------------------------
# Dockerfile for M2TGLGO / ST-image project
#  • CUDA 12.4 runtime & dev
#  • PyTorch 2.5.0 + compatible PyG wheels
#  • All user‑requested Python libraries (pinned where needed)
# ------------------------------------------------------------

# ---------- BASE IMAGE -------------------------------------------------------
# Official PyTorch image with CUDA 12.4 & cuDNN 9 (Python 3.10)
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# ---------- SYSTEM DEPENDENCIES ----------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential gcc g++ git curl ca-certificates \
        libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------- PYTHON PACKAGES ---------------------------------------------------
# 1) Extra wheels index for PyG CUDA 12.4 builds
ENV TORCH_VERSION=2.5.0
ENV CUDA_TAG=cu124
ENV PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

# 2) Upgrade pip & install core deps first to minimise layer invalidation
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3) Install torch‑scatter & torch‑sparse from pre‑built wheels, then PyG core
RUN pip install --no-cache-dir \
        torch-scatter==2.1.2 \
        torch-sparse==0.6.18 \
        -f ${PYG_URL} && \
    pip install --no-cache-dir torch-geometric==2.5.0

# 4) Remaining project libraries (version‑caps from user list)
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    pandas>=1.3.0 \
    scikit-learn>=1.0.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    pillow>=9.0.0 \
    opencv-python-headless>=4.5.0 \
    tensorboard>=2.8.0 \
    wandb>=0.13.0 \
    tqdm>=4.64.0 \
    pyyaml>=6.0 \
    networkx>=2.8.0 \
    scipy>=1.8.0 \
    scanpy>=1.9.0 \
    anndata>=0.8.0 \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    albumentations>=1.3.0 \
    timm>=0.6.0 \
    transformers>=4.20.0 \
    accelerate>=0.20.0

# ---------- DEFAULT WORKDIR --------------------------------------------------
WORKDIR /workspace

# ---------- ENTRYPOINT -------------------------------------------------------
CMD ["/bin/bash"]

