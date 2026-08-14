# comfyui-aws-queue — project notes for Claude

Architecture overview lives in `IMPLEMENTATION_PLAN.md` and `README.md`. This
file captures the parts that are *not* obvious from the code — especially how
custom nodes and models work, and the failure modes we hit getting them
running. Read this before touching the worker image, the custom-node flow, or
the model catalog.

---

## Custom nodes — how they work

**Source of truth: the S3 manifest.** `s3://comfy-outputs-<acct>-<region>/manifests/custom-nodes.json`
lists every pack the deployment should have. The dispatcher Lambda appends to
it when the user clicks Install in the editor's ComfyUI-Manager UI (it
intercepts `POST /manager/queue/install`).

**The common set is BAKED into the worker image at build time.**
- `workers/image/baked_nodes.txt` — the curated pack list (image fleet).
- `workers/video/baked_nodes.txt` — the video fleet's SCAIL-2 pose-control
  packs (KJNodes, WanAnimatePreprocess, SCAIL-pose). The Wan wrapper, VHS and
  Manager are still cloned directly in `workers/video/Dockerfile` and get their
  `.baked` markers there.
- `workers/shared/bake_nodes.py` — runs in `workers/image/Dockerfile` during
  `docker build`: clones + installs each pack, drops a `.baked` marker in it.
- At worker boot `manifest_installer.sync()` reads the manifest; a pack with a
  `.baked` marker is **skipped** (already in the image); only net-new packs
  are git-cloned + pip-installed at runtime.
- Why: installing ~33 packs on every cold boot added ~8 min and crash-looped
  the worker on dependency conflicts. Baking moves that to build time, where
  conflicts fail the *build*, visibly, instead of production.

**The metadata instance is the canonical `/object_info` + extensions
publisher.** Image GPU workers must NOT publish object_info — `worker.py`
skips it for `FLEET == "image"`. A cold worker's node list is partial (custom
nodes still importing) and would overwrite the editor's full one. This was the
cause of nodes repeatedly "disappearing" from the editor.

**To make a custom node usable you need it in three places:** the manifest
(persistence), installed on the metadata instance (editor's node list — needs
a `docker restart comfy-metadata` to load), and installed on the GPU workers
(baked image or runtime manifest).

---

## Custom nodes — dependency-conflict lessons (the hard part)

Custom-node `requirements.txt` files fight over foundational packages and
**will** break the shared Python environment. `manifest_installer.py` defends
against this; understand it before changing the worker image.

| Symptom | Cause | Fix in code |
|---|---|---|
| `numpy.core.multiarray failed to import` — every tensor node breaks | a node's requirements changed numpy | numpy/torch pinned via a pip `--constraint` file (`_constraints_file`) |
| `cv2.dnn has no attribute 'DictValue'` — **ComfyUI crashes entirely on startup** | multiple opencv variants (-python / -headless / -contrib) clobber one `cv2/` dir → mismatched stub vs binary | `_force_clean_opencv()` — uninstall all variants, install one, patch `cv2/typing/__init__.py` |
| Every transformers node IMPORT FAILS (`AutoModel`, `BlipProcessor`…) | a node upgraded transformers to >=4.54, which imports `torch.distributed.tensor.device_mesh` — absent from NGC torch 2.5 | `_force_transformers()` — pin `transformers==4.46.3` |
| `libcudart.so.13: cannot open shared object file` — ComfyUI crash-loop | a node pulled a real `torchaudio` over the Dockerfile's stub; real torchaudio needs a CUDA the image lacks | re-stub torchaudio (Dockerfile RUN after the bake, build-verified) |
| Impact-Pack missing `segment_anything` | `pip install -r requirements.txt` is all-or-nothing — one unsatisfiable line drops every dep | `_pip_install_requirements` falls back to installing line-by-line |

**Key principle:** the foundation repairs (`_force_clean_opencv`,
`_force_transformers`, `_restore_torchaudio_stub`) run as the LAST step of the
bake and of runtime `sync()` — whatever a node did during install, the
foundation is restored afterward. If a node breaks the worker, the cause is
almost always a `requirements.txt` clobbering a foundational package; pin or
repair it rather than fighting the individual node.

The base image is NGC `nvcr.io/nvidia/pytorch` with a custom-built torch —
never let pip replace torch/torchvision/torchaudio (the Dockerfile strips them
from ComfyUI's requirements and stubs torchaudio). numpy must stay `<2`.

---

## Models — how they work

**Storage + mounting.** Models live in S3 at
`s3://comfy-models-<acct>-<region>/<catalog-type>/<filename>`. Workers mount
each type prefix via Mountpoint for S3 (`mount-s3`) at `models/<comfy-dir>/` —
ComfyUI sees them as local files, streamed on first access.

**`workers/shared/model_types.py` `TYPE_DIR`** is the single catalog-type →
ComfyUI-directory map. To support a new model type you MUST add it there (the
worker entrypoint mounts every type in the map) — and mirror it in
`services/downloader/kickoff.py` `ALLOWED_TYPES`.

**The `comfy-models` DynamoDB table is the catalog** (`name`, `type`,
`s3_key`, `size_gb`, …). The editor's model dropdowns come from this catalog,
NOT a filesystem scan: the dispatcher's `/object_info` swaps each model-name
input list for the catalog values (`_scan_catalog_by_type` + the
`_INPUT_TO_MODEL_TYPE` / `_guess_model_type_from_input` mapping). The dropdown
value is the **filename** (basename of `s3_key`).

**To add a model so the editor sees it:**
1. Put the file in S3 at `<type>/<filename>` — `<type>` must be in `model_types.py`.
2. Add a row to the `comfy-models` table.
3. The dispatcher caches `/object_info` ~60 s — the dropdown updates with no restart.

The normal path — `POST /models/download` → download-worker Lambda — does both
S3 + catalog. A file dropped straight into S3 will work for *workers* (they
mount S3) but won't appear in the editor dropdown until it's cataloged. If a
node's input is the ambiguous `model_name`, `_guess_model_type_from_input`
disambiguates by node class (e.g. `UltralyticsDetectorProvider` → `ultralytics`).

**A workflow referencing a model that isn't on the worker fails SILENTLY in a
confusing way.** ComfyUI's `validate_prompt` drops every *output node* whose
input tree contains the missing-model error (`value_not_in_list`) and runs
only the still-valid output nodes — the job reports `status: success` but
produces 0 outputs. Symptom we hit: novaAnimeXL jobs "completed" in ~9 s with
no image because `UpscaleModelLoader` wanted `2x-AnimeSharpV4_RCAN`, absent
from S3 → the whole upscale branch (and the `SaveImageWithMetaData` depending
on it) was dropped, leaving only `PreviewImage`. When a job completes fast
with 0 outputs, check every loader node's model against S3 + the catalog;
ComfyUI `/history` `outputs` shows which nodes actually ran.

---

## Other gotchas

- **Frontend-only nodes** (Notes, rgthree "Fast Groups Bypasser") have no
  `class_type`; ComfyUI's validator rejects the whole prompt over one.
  `comfy_client.submit_prompt` strips them before submitting.
- **`SaveImageWithMetaData` (comfyui_image_metadata_extension) vs ComfyUI's
  async cache.** Its metadata capture introspects ComfyUI's execution cache
  synchronously, but this ComfyUI made `HierarchicalCache.get` a coroutine →
  `'coroutine' object has no attribute 'outputs'` → the exception aborts
  `save_images` and the image never saves. `workers/image/patch_metadata_ext.py`
  (a build-verified Dockerfile step) wraps the `gen_pnginfo()` call so a
  capture failure is non-fatal — the image saves, just without the captured
  A1111 parameter metadata.
- **rgthree's Seed node caps `seed` at 1125899906842624 (2^50).** Randomize
  seeds within that — a larger value fails `prompt_outputs_failed_validation`.
- **API Gateway leaves `%2F` encoded** in `{proxy+}` greedy path params —
  decode it (this broke userdata/workflow save+load).
- **Wildcards** download via the `wildcards` catalog type (a `.zip` pack is
  unzipped by `services/downloader/worker.py`). They are NOT a `models/` type:
  `workers/image/entrypoint.sh` mounts the `wildcards/` S3 prefix at
  Impact-Pack's `custom_wildcards` dir. Impact-Pack scans that dir once at
  startup — a pack downloaded *after* a worker is already running won't be
  seen until the worker is rotated.
- **Worker rotation strands in-flight jobs** as zombie `running` records — when
  rotating the `comfy-image` task definition, expect to clean those up.
- The image fleet is a **spot ASG on 2xlarge (32 GB RAM) instances since
  2026-07-12** — `g5.2xlarge` (A10G, sm_86), `g6.2xlarge` (L4, sm_89), and
  `g6e.xlarge` (L40S, sm_89) — all 24 GB VRAM, 8 vCPU except g6e.xlarge (4
  vCPU/32 GB). Moved off the 16 GB `.xlarge` pools because Z-Image (a
  FLOW-arch model: ~6 GB UNET + ~5.6 GB qwen text encoder + VAE ≈ 12 GB of
  weights + ComfyUI working set) OOM-killed ComfyUI (rc=-9) on a 16 GB
  `.xlarge` even after stripping FaceDetailer/upscale — so FLOW/Flux-class
  jobs, previously unreliable, are now fine (all three pools are ≥32 GB RAM).
  **Primary is `g6.2xlarge` since 2026-08-10** (was `g5.2xlarge`), re-ranked
  on real job-ledger data: 94% of jobs are Z-Image fp8
  (`divingZImageTurbo_v60Fp8`), and Ada (g6.2xlarge/g6e.xlarge) has native
  fp8 support that Ampere (g5.2xlarge) lacks and has to upcast — measured
  median job duration g5.2xlarge 76s vs g6.2xlarge 58s vs g6e.xlarge 47s,
  and at measured spot prices that's ~$0.0234/image (g5.2xlarge) vs
  ~$0.0148/image (g6.2xlarge) vs ~$0.0227/image (g6e.xlarge) — g6.2xlarge is
  both the fastest AND ~35% cheaper per image than either alternative.
  Fallback order is `[g5.2xlarge, g6e.xlarge]`. Allocation strategy is also
  now `price-capacity-optimized` (was `capacity-optimized-prioritized`,
  which under-used the nominal primary: only 5.8% of jobs actually landed on
  it, with 26.9% falling to the ~2x-priced g6e.xlarge).
  **Still NO g4dn** (T4). **Why not g4dn:** a g4dn (T4) `lowest-price` cost
  experiment backfired for the SDXL+ESRGAN-upscale workload (measured
  2026-06-08) — the T4 ran **~192 s/image vs A10G's ~61 s (~3.2× slower)**,
  so at ~$0.24/hr vs ~$0.48/hr it was **~$12.95 vs $8.11 per 1000 images
  (~60% MORE expensive per image)**; and the T4's 16 GB VRAM forced ComfyUI
  to offload ~14 GB to pinned CPU RAM, **OOM-killing ComfyUI on the 16
  GB-RAM g4dn.xlarge** (jobs failed `[Errno 111] Connection refused` to
  `:8188`; the 32 GB g4dn.2xlarge survived but was still slow/pricey).
  xformers efficient attention also fails to load on the worker image (build
  mismatch vs the NGC torch), worsening VRAM pressure. See
  `infra/lib/config.ts` for the canonical config. The us-east-1 G/VT spot
  vCPU quota (raised to **48**, approved 2026-06-08) gates concurrency, but
  `scaling.imageMax` is **3** (3× 8 vCPU 2xlarge = 24 vCPU), leaving most of
  the pool (**shared with the video fleet**) free.
