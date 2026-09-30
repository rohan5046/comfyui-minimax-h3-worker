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

Loaded from the attached network volume, mounted at `/runpod-volume` (see
`extra_model_paths.yaml`). Expected layout — **update this file if your volume's layout
differs**:

```
/runpod-volume/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
/runpod-volume/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
/runpod-volume/models/vae/minimax_h3_video_vae_int8_convrot.safetensors
/runpod-volume/models/vae/minimax_h3_audio_vae_fp32.safetensors
/runpod-volume/models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
```

## Env vars

`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` — same R2
bucket the Next.js app uses (see `.env.example` there).

## Local structure

- `graph_builder.py` — scene → ComfyUI API-format graph, based on `workflows/video_minimax_h3_r2v.json`
  (Comfy-Org's official R2V template).
- `handler.py` — RunPod entrypoint: waits for ComfyUI, downloads reference images, submits
  the graph, uploads the result to R2.
- `Dockerfile` / `start.sh` — bakes ComfyUI + this worker; models come from the volume, not
  the image.
