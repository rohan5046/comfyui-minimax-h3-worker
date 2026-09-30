#!/bin/bash
set -e

# Bridge RunPod's host-cached HuggingFace model (attached via --model-reference,
# not a network volume - removes the single-data-center pin the old volume-mount
# setup had) into the plain paths ComfyUI's loader nodes expect. RunPod's cache
# uses the standard HF layout (models--org--repo/snapshots/<hash>/...); our
# mirror repo (vidGen654/minimax-h3-r2v-comfyui) already uses the same
# diffusion_models/text_encoders/vae/loras subfolder names ComfyUI itself uses,
# so each one is just symlinked wholesale - no per-file mapping needed.
CACHE_BASE=/runpod-volume/huggingface-cache/hub/models--vidGen654--minimax-h3-r2v-comfyui/snapshots
SNAPSHOT_DIR=$(ls -d "$CACHE_BASE"/*/ 2>/dev/null | head -1)
if [ -z "$SNAPSHOT_DIR" ]; then
  # Don't hard-exit here - RunPod's own build-time smoke test boots this
  # container without --model-reference attached (same structural gap the
  # old network-volume setup hit), so a hard exit here fails TEST_FAILED on
  # every build before a real job is ever sent. handler.py's ping
  # short-circuit answers the smoke test fine without touching models; a
  # real job still fails loudly and correctly via its own
  # _wait_for_models_ready() timeout, which names this exact cause.
  echo "WARNING: no cached snapshot found under $CACHE_BASE - was --model-reference attached to this endpoint? Real jobs will fail at _wait_for_models_ready()." >&2
else
  for d in diffusion_models text_encoders vae loras; do
    rm -rf "/comfyui/models/$d"
    ln -s "${SNAPSHOT_DIR}${d}" "/comfyui/models/$d"
  done
fi

python3 /comfyui/main.py \
  --listen 127.0.0.1 \
  --port 8188 \
  --disable-auto-launch &

exec python3 -u /handler.py
