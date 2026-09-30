"""RunPod serverless entrypoint for the MiniMax H3 video endpoint.

Contract is fixed by the app, not this file - see src/lib/runpod.ts
(dispatchJob/buildWorkflowPayload) and src/app/api/webhooks/runpod/route.ts:
  input:  {userId, scene, characterRefs: {label: presigned_get_url}}
  output: {outputStorageKey}  on success
          {error: "..."}      on failure (apply-job-result.ts treats either
                               a non-COMPLETED status OR output.error OR a
                               missing outputStorageKey as failed)

Assumes a ComfyUI server is reachable at COMFYUI_URL - start.sh launches it
as a background process before this process starts (see Dockerfile).
"""
import os
import time
import uuid
import urllib.request
from pathlib import Path

import boto3
import requests
import runpod

from graph_builder import build_scene_graph, SAVE_VIDEO_NODE_ID

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_INPUT_DIR = Path(os.environ.get("COMFYUI_INPUT_DIR", "/comfyui/input"))
COMFYUI_OUTPUT_DIR = Path(os.environ.get("COMFYUI_OUTPUT_DIR", "/comfyui/output"))

READY_TIMEOUT_S = 180  # cold start: image pull is separate; this is ComfyUI's own boot
MODELS_READY_TIMEOUT_S = 120
POLL_INTERVAL_S = 2
JOB_TIMEOUT_S = 1200  # MiniMax H3 video generation is slow - generous ceiling

# Every model file the graph needs, keyed by the loader node class whose
# combo box lists it (folder_paths.get_filename_list() under the hood).
# Confirmed against ComfyUI's own source (nodes.py) that these are classic
# INPUT_TYPES() classmethods re-evaluated fresh on every call - no stale
# caching. Models now arrive via start.sh's symlink bridge from RunPod's
# host-cached HF model (see --model-reference on the endpoint) rather than a
# network volume mount, but this check stays as a cheap readiness gate
# regardless of how the files got there.
REQUIRED_MODELS = {
    "UNETLoader": ("unet_name", ["minimax_h3_ref2va_pruned_int8_convrot.safetensors"]),
    "CLIPLoader": ("clip_name", ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]),
    "VAELoader": (
        "vae_name",
        ["minimax_h3_video_vae_int8_convrot.safetensors", "minimax_h3_audio_vae_fp32.safetensors"],
    ),
    "LoraLoaderModelOnly": ("lora_name", ["minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"]),
}


def _wait_for_comfyui_ready() -> None:
    deadline = time.time() + READY_TIMEOUT_S
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException as exc:
            last_err = exc
        time.sleep(1)
    raise RuntimeError(f"ComfyUI did not become ready within {READY_TIMEOUT_S}s: {last_err}")


def _wait_for_models_ready() -> None:
    deadline = time.time() + MODELS_READY_TIMEOUT_S
    last_missing: dict[str, list[str]] = {}
    while time.time() < deadline:
        last_missing = {}
        for node_class, (input_key, expected_files) in REQUIRED_MODELS.items():
            try:
                r = requests.get(f"{COMFYUI_URL}/object_info/{node_class}", timeout=10)
                r.raise_for_status()
                options = r.json()[node_class]["input"]["required"][input_key][0]
            except (requests.RequestException, KeyError, ValueError):
                last_missing[node_class] = expected_files
                continue
            missing = [f for f in expected_files if f not in options]
            if missing:
                last_missing[node_class] = missing
        if not last_missing:
            return
        time.sleep(POLL_INTERVAL_S)
    raise RuntimeError(
        f"Model files still not visible to ComfyUI after {MODELS_READY_TIMEOUT_S}s "
        f"(check start.sh's symlink bridge ran and the HF model-reference cache is populated): {last_missing}"
    )


def _download_character_refs(character_refs: dict) -> dict:
    """Downloads each {label: presigned_get_url} into ComfyUI's input/ dir.
    Returns {label: filename} for graph_builder.build_scene_graph - LoadImage
    resolves filenames relative to input/, it does not take URLs."""
    COMFYUI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    filenames: dict[str, str] = {}
    for label, url in (character_refs or {}).items():
        filename = f"ref_{uuid.uuid4().hex}.png"
        dest = COMFYUI_INPUT_DIR / filename
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - url is our own R2 presigned GET
        filenames[label] = filename
    return filenames


def _queue_prompt(graph: dict) -> str:
    r = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": graph}, timeout=30)
    # ComfyUI answers a validation failure with HTTP 400 and the actual
    # reason in the body ({"error": {...}, "node_errors": {...}}) - reading
    # the body BEFORE raise_for_status() is what surfaces that reason
    # instead of a useless generic "400 Client Error".
    try:
        body = r.json()
    except ValueError:
        body = None
    if not r.ok:
        detail = body if body is not None else r.text
        raise RuntimeError(f"ComfyUI rejected the workflow ({r.status_code}): {detail}")
    if body.get("error"):
        raise RuntimeError(f"ComfyUI rejected the workflow: {body['error']}")
    return body["prompt_id"]


def _wait_for_completion(prompt_id: str) -> dict:
    deadline = time.time() + JOB_TIMEOUT_S
    while time.time() < deadline:
        r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        r.raise_for_status()
        history = r.json()
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI execution failed: {status}")
            if status.get("completed"):
                return entry
        time.sleep(POLL_INTERVAL_S)
    raise RuntimeError(f"Timed out after {JOB_TIMEOUT_S}s waiting for ComfyUI")


def _extract_video_file(history_entry: dict) -> Path:
    outputs = history_entry.get("outputs", {}).get(SAVE_VIDEO_NODE_ID, {})
    # SaveVideo (comfy_api/latest/_ui.py, class PreviewVideo.as_dict) returns
    # {"images": [...], "animated": (True,)} - confirmed against ComfyUI's
    # own source, not the SaveImage node's key by coincidence: SaveVideo
    # reuses it deliberately so older frontends that only know "images"
    # still render it.
    items = outputs.get("images")
    if items:
        item = items[0]
        return COMFYUI_OUTPUT_DIR / item.get("subfolder", "") / item["filename"]
    raise RuntimeError(f"No video found in history output for node {SAVE_VIDEO_NODE_ID}: {outputs}")


def _upload_to_r2(local_path: Path, user_id: str) -> str:
    key = f"users/{user_id}/outputs/{uuid.uuid4().hex}.mp4"
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    client.upload_file(
        str(local_path), os.environ["R2_BUCKET_NAME"], key, ExtraArgs={"ContentType": "video/mp4"}
    )
    return key


def handler(job: dict) -> dict:
    job_input = job["input"]
    # Platform smoke-test hook (see .runpod/tests.json): kept as a fast,
    # cheap "does the container boot and answer a job" check distinct from
    # a real job's full _wait_for_models_ready() + generation path - the
    # test harness may not attach the model-reference cache the same way a
    # real endpoint deploy does.
    if job_input.get("ping"):
        return {"pong": True}

    user_id = job_input["userId"]
    scene = job_input["scene"]
    character_refs = job_input.get("characterRefs") or {}

    started = time.time()
    try:
        _wait_for_comfyui_ready()
        _wait_for_models_ready()
        character_filenames = _download_character_refs(character_refs)
        graph = build_scene_graph(scene, character_filenames)
        prompt_id = _queue_prompt(graph)
        history_entry = _wait_for_completion(prompt_id)
        video_path = _extract_video_file(history_entry)
        output_key = _upload_to_r2(video_path, user_id)
    except Exception as exc:  # noqa: BLE001 - the app's webhook contract
        # wants a string in output.error, not an unhandled-exception job
        # failure (apply-job-result.ts checks output?.error explicitly).
        return {"error": str(exc)}

    return {
        "outputStorageKey": output_key,
        "comfyExecMs": round((time.time() - started) * 1000),
    }


runpod.serverless.start({"handler": handler})
