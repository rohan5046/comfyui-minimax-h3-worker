# Video endpoint worker: ComfyUI + MiniMax H3 Reference-to-Video, models
# mounted from the network volume at /runpod-volume (see
# extra_model_paths.yaml), everything else baked in (golden-path "bake code,
# mount data" split).
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg libgl1 \
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
# cu128 build - installing plain `torch` before requirements.txt let pip
# pull in an unpinned, mismatched torchvision afterward, which crashed
# ComfyUI at import time with "operator torchvision::nms does not exist"
# (an ABI mismatch, not a missing package - confirmed live on the Hub's
# test run, A100 80GB, 2026-09-29). Installing the matched triple last
# guarantees it's the final, authoritative state regardless of what
# requirements.txt pulled in.
RUN pip install --no-cache-dir --force-reinstall \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /worker-requirements.txt
RUN pip install --no-cache-dir -r /worker-requirements.txt

COPY extra_model_paths.yaml /extra_model_paths.yaml
COPY graph_builder.py handler.py /
COPY workflows /workflows
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
