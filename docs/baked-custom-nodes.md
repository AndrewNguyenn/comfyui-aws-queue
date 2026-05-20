# Custom nodes to bake into the worker image

## Why

Custom node packs currently live in the S3 manifest (`manifests/custom-nodes.json`).
Every GPU worker, on cold start, re-clones and `pip install`s **all** of them
before ComfyUI can take a job. With 27 packs that adds **~5–10 minutes to every
cold start**.

Baking the common packs into the worker Docker image (cloned + deps installed
at image-build time) removes that from the hot path — a cold worker just starts
ComfyUI. The manifest still handles anything installed *after* the image was
built, so the "install a node and it persists" flow is unchanged.

> ComfyUI-Manager is already in the image — not listed below.

---

## Tier 1 — currently installed (your 27 manifest packs)

Bake all of these in first.

| Pack | Repo |
|------|------|
| ComfyUI-Impact-Pack | https://github.com/ltdrdata/ComfyUI-Impact-Pack |
| ComfyUI-Impact-Subpack | https://github.com/ltdrdata/ComfyUI-Impact-Subpack |
| rgthree-comfy | https://github.com/rgthree/rgthree-comfy |
| ComfyUI-Custom-Scripts | https://github.com/pythongosssss/ComfyUI-Custom-Scripts |
| ComfyUI-KJNodes | https://github.com/kijai/ComfyUI-KJNodes |
| ComfyUI-Easy-Use | https://github.com/yolain/ComfyUI-Easy-Use |
| ComfyUI-Florence2 | https://github.com/kijai/ComfyUI-Florence2 |
| ComfyUI_UltimateSDUpscale | https://github.com/ssitu/ComfyUI_UltimateSDUpscale |
| ComfyUI-WD14-Tagger | https://github.com/pythongosssss/ComfyUI-WD14-Tagger |
| RES4LYF | https://github.com/ClownsharkBatwing/RES4LYF |
| ComfyUI_essentials | https://github.com/cubiq/ComfyUI_essentials |
| ComfyUI-Image-Saver | https://github.com/alexopus/ComfyUI-Image-Saver |
| comfy-image-saver | https://github.com/giriss/comfy-image-saver |
| Derfuu_ComfyUI_ModdedNodes | https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes |
| ComfyUI-LogicUtils | https://github.com/aria1th/ComfyUI-LogicUtils |
| cg-use-everywhere | https://github.com/chrisgoringe/cg-use-everywhere |
| virtuoso-nodes | https://github.com/chrisfreilich/virtuoso-nodes |
| ComfyUI-mxToolkit | https://github.com/Smirnov75/ComfyUI-mxToolkit |
| was-node-suite-comfyui | https://github.com/ltdrdata/was-node-suite-comfyui |
| ComfyUI_Comfyroll_CustomNodes | https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes |
| ComfyUI-AdvancedLivePortrait | https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait |
| ComfyUI-LevelPixel | https://github.com/LevelPixel/ComfyUI-LevelPixel |
| comfyui_image_metadata_extension | https://github.com/edelvarden/comfyui_image_metadata_extension |
| ComfyUI_LayerStyle | https://github.com/chflame163/ComfyUI_LayerStyle |
| ComfyUI-post-processing-nodes | https://github.com/EllangoK/ComfyUI-post-processing-nodes |
| ComfyUI-Crystools | https://github.com/crystian/ComfyUI-Crystools |
| Civicomfy | https://github.com/MoonGoblinDev/Civicomfy |

---

## Tier 2 — most-used on CivitAI, not yet installed

The packs that show up most often in shared CivitAI workflows (SDXL / Pony /
Illustrious / Flux image generation). Strong candidates to add.

| Pack | What it's for | Repo |
|------|---------------|------|
| ComfyUI_IPAdapter_plus | IP-Adapter — style / character / face reference | https://github.com/cubiq/ComfyUI_IPAdapter_plus |
| comfyui_controlnet_aux | ControlNet preprocessors (depth, pose, canny, …) | https://github.com/Fannovel16/comfyui_controlnet_aux |
| ComfyUI-Advanced-ControlNet | Advanced ControlNet scheduling / masking | https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet |
| ComfyUI-Inspire-Pack | Companion to Impact-Pack — prompts, regional, seeds | https://github.com/ltdrdata/ComfyUI-Inspire-Pack |
| efficiency-nodes-comfyui | Efficiency loaders / XY-plot | https://github.com/jags111/efficiency-nodes-comfyui |
| ComfyUI-GGUF | GGUF-quantized models (Flux on smaller VRAM) | https://github.com/city96/ComfyUI-GGUF |
| ComfyUI_FizzNodes | Prompt scheduling / interpolation | https://github.com/FizzleDorf/ComfyUI_FizzNodes |
| ComfyUI-Detail-Daemon | Detail enhancement during sampling | https://github.com/Jonseed/ComfyUI-Detail-Daemon |
| ComfyUI-tooling-nodes | API-friendly helper nodes | https://github.com/Acly/comfyui-tooling-nodes |
| comfyui-art-venture | Misc utility / conditioning nodes | https://github.com/sipherxyz/comfyui-art-venture |

## Tier 3 — popular for video / animation (add if you do video)

| Pack | What it's for | Repo |
|------|---------------|------|
| ComfyUI-AnimateDiff-Evolved | AnimateDiff motion modules | https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved |
| ComfyUI-VideoHelperSuite | Video load / combine / preview | https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite |
| ComfyUI-Frame-Interpolation | RIFE / FILM frame interpolation | https://github.com/Fannovel16/ComfyUI-Frame-Interpolation |
| ComfyUI-ReActor | Face swap (maintained fork) | https://github.com/Gourieff/ComfyUI-ReActor |

---

## Implementation notes (for later)

- Bake into the **worker** image (`workers/image/Dockerfile`) — clone each repo
  into `custom_nodes/` and `pip install -r requirements.txt` at build time.
- Pin each pack to a **commit SHA**, not a moving branch — a surprise upstream
  change shouldn't break a worker build.
- The boot-time `manifest_installer` should **skip** a pack that already exists
  in `custom_nodes/` (already baked) so it doesn't re-clone — keeps the
  "install later, persists" flow working for anything added post-build.
- Some packs pull heavy deps (onnxruntime, mediapipe, opencv) — watch the image
  size; the video fleet's 250 GB root volume has room, image fleet has 150 GB.
- Tier 1 is the priority; Tier 2/3 are optional and grow the image.
