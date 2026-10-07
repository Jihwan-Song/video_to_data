# SAM 3 needs PyTorch >= 2.7 and CUDA >= 12.6, newer than the 2.5.1 / 12.4 base
# of the other modules.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY v2d_common /workspace/v2d_common
COPY v2d_sam3 /workspace/v2d_sam3

RUN pip install -e "/workspace/v2d_common[io]" -e /workspace/v2d_sam3/lib
