"""Turns a PublicVideoScene (src/lib/scene-validation.ts) into a ComfyUI
API-format prompt graph for MiniMax H3 Reference-to-Video.

Base graph is workflows/video_minimax_h3_r2v.json - Comfy-Org's official R2V
template (node ids below refer to that file). Node 136
(MiniMaxH3ReferenceToVideo) takes reference images as an Autogrow input,
tagged <Picture i> (1-based) in the prompt text - see
https://docs.comfy.org/built-in-nodes/MiniMaxH3ReferenceToVideo. The stock
template ships with none wired in (it's a text-only example), so those inputs
+ their LoadImage nodes are added here per job.

IMPORTANT - API-format key syntax for this Autogrow input (confirmed against
Comfy-Org/ComfyUI issue #15667, closed as "reporter error" but with the real
answer in the resolving comment): in an API-format /prompt graph these are
FLAT, DOTTED, and 0-INDEXED keys - "ref_images.ref_image_0",
"ref_images.ref_image_1", ... - not a nested dict ({"ref_image_1": ...}, which
ComfyUI's execution.py silently drops with no error - looks conditioned,
isn't) and not flat 1-indexed keys (ref_image_1, ..., which a newer ComfyUI
rejects outright with "execute() got an unexpected keyword argument
'ref_image_1'" - confirmed live via a real failed job on this endpoint,
2026-09-30). ComfyUI's own execution.py regroups dotted keys through
_io.build_nested_inputs() before calling execute() - this is the only form
that's both accepted AND actually affects the output.
"""
import copy
import json
import random
from pathlib import Path

WORKFLOW_PATH = Path(__file__).parent / "workflows" / "video_minimax_h3_r2v.json"
_BASE_WORKFLOW = json.loads(WORKFLOW_PATH.read_text())

# ResolutionSelector's exact combo strings (node 115) - confirmed against
# docs.comfy.org/built-in-nodes/ResolutionSelector. The app only ever sends
# the short labels from VIDEO_ASPECT_RATIOS
# (src/components/console/create/types.ts); "Auto" has no ResolutionSelector
# equivalent, so it falls back to the template's own default.
ASPECT_RATIO_MAP = {
    "1:1": "1:1 (Square)",
    "21:9": "21:9 (Ultrawide)",
    "16:9": "16:9 (Widescreen)",
    "9:16": "9:16 (Portrait Widescreen)",
    "3:4": "3:4 (Portrait Standard)",
    "4:3": "4:3 (Standard)",
}
DEFAULT_ASPECT_RATIO = "16:9 (Widescreen)"

# ResolutionSelector's `megapixels` param (node 115) - independent of
# aspect_ratio, this is the actual resolution-tier control. Values chosen to
# match real 16:9 pixel counts for these tiers (854x480 ~= 0.41MP rounds to
# 0.4; 1366x768 ~= 1.05MP rounds to 1.0, which also happens to be this
# node's own shipped default per comfy_extras/nodes_resolution.py). The app
# only ever sends "480p"/"768p" (src/components/console/create/types.ts).
RESOLUTION_MEGAPIXELS_MAP = {
    "480p": 0.4,
    "768p": 1.0,
}
DEFAULT_MEGAPIXELS = 1.0

MAX_REF_IMAGES = 9

SAVE_VIDEO_NODE_ID = "92"
PROMPT_NODE_ID = "138"
RESOLUTION_NODE_ID = "115"
DURATION_NODE_ID = "132"
SEED_NODE_ID = "129"
REFERENCE_NODE_ID = "136"


def _build_prompt(scene: dict, ordered_labels: list[str]) -> str:
    """Composes the free-text prompt. Reference tags are listed up front so
    the model has an explicit label -> <Picture i> mapping, then the scene's
    own description/camera/audio guidance follows."""
    lines = []
    if ordered_labels:
        tags = ", ".join(f"<Picture {i + 1}> = {label}" for i, label in enumerate(ordered_labels))
        lines.append(f"Reference images: {tags}.")
    action = (scene.get("action") or "").strip()
    if action:
        lines.append(action)
    camera_custom = (scene.get("camera_custom") or "").strip()
    if camera_custom:
        lines.append(f"Camera: {camera_custom}")
    audio_tag = (scene.get("audio_tag") or "").strip()
    if audio_tag:
        lines.append(f"Audio: {audio_tag}")
    return "\n\n".join(lines)


def build_scene_graph(scene: dict, character_image_filenames: dict) -> dict:
    """
    scene: the PublicVideoScene dict (see src/lib/scene-validation.ts).
    character_image_filenames: {label: filename}, where filename is a file
      already saved under ComfyUI's input/ directory (LoadImage's "image"
      widget takes a filename it resolves relative to input/, never a URL -
      handler.py downloads each characterRefs[label] presigned URL there
      before calling this function).

    Returns a fresh API-format prompt graph (dict of node id -> node),
    independent of _BASE_WORKFLOW (deep-copied, safe to mutate per job).
    """
    graph = copy.deepcopy(_BASE_WORKFLOW)

    # Only characters actually named in scene["characters"] AND present in
    # character_image_filenames get wired in - order follows
    # scene["characters"] so the <Picture i> tags in the composed prompt
    # line up with the ref_image_i slots below.
    ordered_labels = [
        label for label in scene.get("characters", []) if label in character_image_filenames
    ]
    if len(ordered_labels) > MAX_REF_IMAGES:
        raise ValueError(
            f"MiniMax H3 Reference-to-Video supports at most {MAX_REF_IMAGES} reference "
            f"images, got {len(ordered_labels)}"
        )

    next_node_id = max(int(node_id) for node_id in graph) + 1
    for i, label in enumerate(ordered_labels):
        load_image_id = str(next_node_id)
        next_node_id += 1
        graph[load_image_id] = {
            "inputs": {"image": character_image_filenames[label]},
            "class_type": "LoadImage",
            "_meta": {"title": f"Load Image - {label}"},
        }
        graph[REFERENCE_NODE_ID]["inputs"][f"ref_images.ref_image_{i}"] = [load_image_id, 0]

    graph[PROMPT_NODE_ID]["inputs"]["value"] = _build_prompt(scene, ordered_labels)

    aspect_ratio = scene.get("aspect_ratio") or "Auto"
    graph[RESOLUTION_NODE_ID]["inputs"]["aspect_ratio"] = ASPECT_RATIO_MAP.get(
        aspect_ratio, DEFAULT_ASPECT_RATIO
    )

    resolution = scene.get("resolution")
    graph[RESOLUTION_NODE_ID]["inputs"]["megapixels"] = RESOLUTION_MEGAPIXELS_MAP.get(
        resolution, DEFAULT_MEGAPIXELS
    )

    duration = scene.get("duration") or 5
    graph[DURATION_NODE_ID]["inputs"]["value"] = float(duration)

    # A fresh seed per job - MiniMax H3 has no meaningful "reuse seed" UX in
    # the app today, so this is unconditional.
    graph[SEED_NODE_ID]["inputs"]["noise_seed"] = random.randint(0, 2**32 - 1)

    return graph
