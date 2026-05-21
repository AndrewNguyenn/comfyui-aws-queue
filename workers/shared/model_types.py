"""Single source of truth for the catalog-type → ComfyUI-directory mapping.

Used by:
  - workers/shared/warm_pinned.py (where pinned files appear in the FUSE mount)
  - workers/shared/worker.py (model-name extraction)
  - workers/image/entrypoint.sh + workers/video/entrypoint.sh (which S3
    prefixes to mount, at which paths) via the `python3 -m model_types --bash`
    emit below

The mapping is also mirrored in services/downloader/kickoff.py ALLOWED_TYPES
(can't import worker code from Lambda; keep both in sync when adding types).
"""

from __future__ import annotations

# Catalog type → ComfyUI models/<dir>/. Every catalog type we ever write to S3
# must appear here. Types whose catalog name matches the dir name still need
# an entry so the entrypoint knows to mount them.
TYPE_DIR: dict[str, str] = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "clip": "clip",
    "clip_vision": "clip_vision",
    "embedding": "embeddings",
    "upscale": "upscale_models",
    "diffusion_models": "diffusion_models",
    "text_encoders": "text_encoders",
    "style_models": "style_models",
    "gligen": "gligen",
    "hypernetworks": "hypernetworks",
    "photomaker": "photomaker",
    "audio_encoders": "audio_encoders",
    "vae_approx": "vae_approx",
    "model_patches": "model_patches",
    "unet": "unet",
    # Impact-Pack / Impact-Subpack detector models (face_yolov8m-seg etc.).
    # ComfyUI reads models/ultralytics/{bbox,segm}/.
    "ultralytics": "ultralytics",
}


def comfy_dir(catalog_type: str) -> str:
    """Return the ComfyUI models/ subdir for a catalog type. Defaults to the
    type name itself for unknown types (matches ComfyUI's folder_paths
    fallback)."""
    return TYPE_DIR.get(catalog_type, catalog_type)


def _emit_bash() -> None:
    """Print the mapping as a bash associative-array body so entrypoint.sh can
    eval it. Output is exactly:
        [checkpoint]=checkpoints
        [lora]=loras
        ...
    """
    for k, v in TYPE_DIR.items():
        print(f"[{k}]={v}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--bash":
        _emit_bash()
    else:
        for k, v in TYPE_DIR.items():
            print(f"{k} -> {v}")
