# Video endpoint worker: ComfyUI + MiniMax H3 Reference-to-Video, models now
# baked directly into the image (matching the sibling flux2-klein-worker),
# NOT mounted from a network volume anymore. The volume-mounted setup pinned
# this endpoint to a single data center (US-IL-1, where the volume lives) -
# network volumes are physically tied to one region, so RunPod's scheduler
# could never fail over anywhere else when that region's ADA_24 pool ran
# low on stock (confirmed live: "endpoint not found"/no-capacity failures
# traced to exactly this). Baking the ~39GB of weights into the image
# instead removes that region lock entirely - the endpoint can schedule on
# any data center with free capacity, same as the image worker already
# does, at the cost of a much larger one-time image build/push.
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

# Model files baked directly into ComfyUI's own default models/ tree - no
# extra_model_paths.yaml, no network volume. Source: Comfy-Org/MiniMax-H3 on
# HuggingFace (the same official org that published flux2-klein's files for
# the sibling worker) - confirmed this exact filename/subpath layout via
# that repo's own file listing, and every URL confirmed public/ungated
# (anonymous HTTP 200) before committing to this Dockerfile, same discipline
# as the image worker.
#
# Split into one RUN per file (not one chained RUN) deliberately: the
# previous attempt at this failed fast with a generic docker exit-1 and no
# retrievable logs. One RUN per ~GB-scale download means RunPod's build
# layer cache (--cache-from/--cache-to, confirmed present in the build
# command) can resume from whichever file succeeded last time instead of
# re-downloading all ~39GB on every retry - and if a specific file is the
# actual problem, it fails in isolation instead of inside one large opaque
# step. --tries/--waitretry guard against a transient network blip on a
# multi-GB transfer; -nv (not -q) keeps a size/rate summary line instead of
# fully silent output, in case fuller build logs become inspectable later.
RUN mkdir -p models/diffusion_models models/text_encoders models/vae models/loras
RUN wget -nv --tries=3 --waitretry=10 -O models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
RUN wget -nv --tries=3 --waitretry=10 -O models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
RUN wget -nv --tries=3 --waitretry=10 -O models/vae/minimax_h3_video_vae_int8_convrot.safetensors \
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors"
RUN wget -nv --tries=3 --waitretry=10 -O models/vae/minimax_h3_audio_vae_fp32.safetensors \
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors"
RUN wget -nv --tries=3 --waitretry=10 -O models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

COPY graph_builder.py handler.py /
COPY workflows /workflows
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
