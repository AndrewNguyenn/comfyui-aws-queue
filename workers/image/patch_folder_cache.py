"""Build-time patch — give ComfyUI's model-folder list cache a 60 s TTL.

ComfyUI's `folder_paths.cached_filename_list_` only invalidates the cached
model list when a model folder's mtime changes. The worker's model dirs are
mount-s3 (FUSE) mounts, and mount-s3 does NOT bump a directory's mtime when a
new object appears under its S3 prefix — so a model added to S3 while a
worker is already running stays invisible to ComfyUI until the process
restarts. ComfyUI then fails prompt validation (`value_not_in_list`) and
silently drops the SaveImage node → the job "completes" with 0 outputs.

Fix: add a time-based TTL. `cached_filename_list_`'s cache tuple carries the
scan time (`out[2]`, a `time.perf_counter()` value); if the cached list is
older than 60 s, force a rescan. mount-s3's own `--metadata-ttl 60` means a
model added to S3 becomes visible to ComfyUI within ~2 minutes — no restart.

Runs in the Dockerfile; exits non-zero (fails the build) if the anchor is
missing or the result doesn't parse.
"""

import ast
import pathlib
import sys

FP = pathlib.Path("/opt/comfy/folder_paths.py")
ANCHOR = "    out = filename_list_cache[folder_name]\n"
REPLACEMENT = (
    "    out = filename_list_cache[folder_name]\n"
    "    # TTL (patched) — mount-s3 never bumps a dir's mtime when a model is\n"
    "    # added to its S3 prefix, so the mtime check below never fires.\n"
    "    # Force a rescan if the cached list is older than 60 s.\n"
    "    if time.perf_counter() - out[2] > 60.0:\n"
    "        return None\n"
)

if not FP.is_file():
    print("patch_folder_cache: /opt/comfy/folder_paths.py not found", flush=True)
    sys.exit(1)

txt = FP.read_text()
if "TTL (patched)" in txt:
    print("patch_folder_cache: already patched", flush=True)
    sys.exit(0)
if ANCHOR not in txt:
    print("patch_folder_cache: ANCHOR NOT FOUND — folder_paths.py changed upstream",
          flush=True)
    sys.exit(1)

patched = txt.replace(ANCHOR, REPLACEMENT, 1)
ast.parse(patched)  # build fails here if the patch produced invalid Python
FP.write_text(patched)
print("patch_folder_cache: patched folder_paths.py — 60 s model-list cache TTL",
      flush=True)
