# Video endpoint worker: ComfyUI + MiniMax H3 Reference-to-Video, models
# mounted from the network volume at /runpod-volume (see
# extra_model_paths.yaml), everything else baked in (golden-path "bake code,
# mount data" split).
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
# build-essential (gcc/g++/make) + python3-dev: the cu130 torch build (see
# below) has a newer torch._native ops registry that JIT-compiles some
# kernels via Triton at first use (confirmed live, RTX 4090, 2026-09-30:
# MiniMax H3's Qwen3-VL text encoder failed the CLIP-encode step with
# "Failed to find C compiler" while computing rotary embeddings) - not
# something the cu128 build on this same image ever hit, so easy to miss
# until you exercise the actual text-encoding path, not just VAE/model
# loading. build-essential alone wasn't enough (confirmed live, A100 80GB,
# 2026-09-30): gcc ran but failed compiling Triton's driver.c with exit
# status 1 - that file #includes Python.h, which python3-dev provides and
# a bare python3/python3-pip install doesn't.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ffmpeg libgl1 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pinned to `master`, not a release tag: MiniMaxH3ReferenceToVideo,
# ResolutionSelector, ComfyMathExpression, ComfySwitchNode, CreateVideo,
# SaveVideo and VAEDecodeAudio are all recent core-node additions (2026) -
# no tagged release was confirmed to include all of them as of this build.
# Re-pin to a specific commit SHA once this image is verified working, so a
# future upstream change can't silently break the graph.
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /comfyui
WORKDIR /comfyui
RUN pip install --no-cache-dir -r requirements.txt
# torch/torchvision/torchaudio installed together, LAST, pinned to the same
# build - installing plain `torch` before requirements.txt let pip pull in
# an unpinned, mismatched torchvision afterward, which crashed ComfyUI at
# import time with "operator torchvision::nms does not exist" (an ABI
# mismatch, not a missing package - confirmed live, A100 80GB, 2026-09-29).
# Installing the matched triple last guarantees it's the final,
# authoritative state regardless of what requirements.txt pulled in.
#
# cu130, not cu128: a real generation job on the deployed endpoint (RTX
# 4090, 2026-09-30) failed mid-VAE-decode with "detect_k_anchor kernel
# launch failed: CUDA driver version is insufficient for CUDA runtime
# version" inside comfy_kitchen's int8_attention (MiniMax H3's quantized
# VAE attention). ComfyUI's own boot log had already been warning
# "WARNING: You need pytorch with cu130 or higher to use optimized CUDA
# operations" on every prior run - this is that warning made real, not a
# benign notice. comfy_kitchen ships prebuilt kernels that need a cu130+
# runtime regardless of GPU generation; confirmed torch/torchvision/
# torchaudio 2.11.0 all publish cu130 cp310 manylinux wheels before making
# this change.
RUN pip install --no-cache-dir --force-reinstall \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

COPY requirements.txt /worker-requirements.txt
RUN pip install --no-cache-dir -r /worker-requirements.txt

COPY extra_model_paths.yaml /extra_model_paths.yaml
COPY graph_builder.py handler.py /
COPY workflows /workflows
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
