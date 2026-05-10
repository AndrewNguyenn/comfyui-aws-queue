"""
Heuristic workflow type classification.

Walks a ComfyUI API-format workflow JSON and decides whether it's an image or
video generation workflow based on the node class_types present.

The frontend SHOULD send `type: "image"|"video"` explicitly; this is a fallback
for legacy clients or sanity checking.
"""

from __future__ import annotations

# Patterns that strongly indicate a video workflow. Any match → video.
VIDEO_NODE_PATTERNS = (
    "WanVideo",
    "Wan2",
    "Wan_",
    "HunyuanVideo",
    "LTXVideo",
    "CogVideoX",
    "AnimateDiff",
    "VHS_VideoCombine",
    "VideoLinearCFG",
    "SVD_",
    "StableVideo",
    "Mochi",
    "Pyramid",
)


def classify_workflow(workflow: dict) -> str:
    """Returns 'image' or 'video'. Defaults to 'image' on no match."""
    for _node_id, node in workflow.items():
        class_type = ""
        if isinstance(node, dict):
            class_type = node.get("class_type", "")
        if not isinstance(class_type, str):
            continue
        for pattern in VIDEO_NODE_PATTERNS:
            if pattern in class_type:
                return "video"
    return "image"
