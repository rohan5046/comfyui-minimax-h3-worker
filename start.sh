#!/bin/bash
set -e

python3 /comfyui/main.py \
  --listen 127.0.0.1 \
  --port 8188 \
  --disable-auto-launch &

exec python3 -u /handler.py
