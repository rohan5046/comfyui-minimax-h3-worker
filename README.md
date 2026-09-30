# MiniMax H3 Reference-to-Video worker

RunPod serverless worker: ComfyUI + [MiniMax H3 Reference-to-Video](https://docs.comfy.org/built-in-nodes/MiniMaxH3ReferenceToVideo).
Deployed by [publicapp](../publicapp)'s `RUNPOD_ENDPOINT_ID_VIDEO` — see `src/lib/runpod.ts`
in that repo for the dispatcher this contract must match.

## Job contract

**Input** (`{"input": {...}}`):

```json
{
  "userId": "string",
  "scene": {
    "id": "string", "camera": "custom", "camera_custom": "string",
    "characters": ["label", "..."], "continue_from_previous": false,
    "continuity": "string", "duration": 5, "action": "string",
    "dialogue": [], "audio_tag": "string", "aspect_ratio": "16:9"
  },
  "characterRefs": { "label": "https://presigned-get-url" }
}
```

`scene.characters` order determines `<Picture i>` reference order (1-based); a label with
no matching entry in `characterRefs` is skipped. `aspect_ratio` accepts `Auto`, `1:1`,
`21:9`, `16:9`, `9:16`, `3:4`, `4:3` (see `graph_builder.py`'s `ASPECT_RATIO_MAP`).

**Output:**

```json
{ "outputStorageKey": "users/<userId>/outputs/<uuid>.mp4", "comfyExecMs": 12345 }
```

or `{ "error": "..." }` on failure.

## Models

Attached via RunPod's host-cached HuggingFace model feature (`--model-reference` on the
endpoint, pointed at [`vidGen654/minimax-h3-r2v-comfyui`](https://huggingface.co/vidGen654/minimax-h3-r2v-comfyui) -
a small mirror of just the 5 files this worker needs, since the official
`Comfy-Org/MiniMax-H3` repo bundles every quantization variant together at ~489GB and
RunPod's caching works at whole-repo granularity, not per-file).

RunPod caches the repo host-side in the standard HF layout under
`/runpod-volume/huggingface-cache/hub/models--vidGen654--minimax-h3-r2v-comfyui/snapshots/<hash>/`.
`start.sh` symlinks each of that snapshot's subfolders (`diffusion_models/`,
`text_encoders/`, `vae/`, `loras/` - the mirror repo uses the same names ComfyUI itself
does) directly into `/comfyui/models/`, before ComfyUI starts:

```
diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_int8_convrot.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
```

This replaced an earlier network-volume-mounted setup - volumes are pinned to a single
data center, which caused real dispatch failures once that region's GPU pool ran low on
stock. Host-cached models carry no such region lock (confirmed: `--model-reference`
requires a public repo on this account today - private-repo validation doesn't correctly
read the endpoint's `HF_TOKEN`, reported to RunPod - so the mirror repo above is public,
same as the files it copies).

## Env vars

`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` — same R2
bucket the Next.js app uses (see `.env.example` there).

## Local structure

- `graph_builder.py` — scene → ComfyUI API-format graph, based on `workflows/video_minimax_h3_r2v.json`
  (Comfy-Org's official R2V template).
- `handler.py` — RunPod entrypoint: waits for ComfyUI, downloads reference images, submits
  the graph, uploads the result to R2.
- `Dockerfile` — bakes ComfyUI + this worker; models are neither baked nor
  volume-mounted, see "Models" above.
- `start.sh` — the model-cache symlink bridge, then launches ComfyUI + `handler.py`.
