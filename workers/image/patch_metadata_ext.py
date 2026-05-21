"""Build-time patch — make SaveImageWithMetaData's metadata capture non-fatal.

comfyui_image_metadata_extension's SaveImageWithMetaData builds PNG metadata by
introspecting ComfyUI's execution cache. This ComfyUI version made that cache
async (HierarchicalCache.get is a coroutine); the extension calls it
synchronously → "'coroutine' object has no attribute 'outputs'" → the
exception propagates out of save_images and the IMAGE NEVER SAVES.

Wrap the gen_pnginfo() call so a capture failure logs and falls back to empty
metadata — the image still saves. Runs in the Dockerfile after the bake;
exits non-zero (fails the build) if the anchor is missing or the result
doesn't parse, so a broken patch is caught at build time, not in production.
"""

import ast
import pathlib
import sys

NODE = pathlib.Path(
    "/opt/comfy/custom_nodes/comfyui_image_metadata_extension"
    "/modules/nodes/node.py"
)
ANCHOR = "pnginfo_dict = pnginfo_dict or self.gen_pnginfo(prompt, prefer_nearest)"
REPLACEMENT = (
    "try:\n"
    "                pnginfo_dict = pnginfo_dict or self.gen_pnginfo(prompt, prefer_nearest)\n"
    "            except Exception as _meta_e:\n"
    "                print('[image_metadata_extension] metadata capture failed:', repr(_meta_e))\n"
    "                pnginfo_dict = pnginfo_dict or {}"
)

if not NODE.is_file():
    print(f"patch_metadata_ext: {NODE} not found — skipping", flush=True)
    sys.exit(0)

txt = NODE.read_text()
if "metadata capture failed" in txt:
    print("patch_metadata_ext: already patched", flush=True)
    sys.exit(0)
if ANCHOR not in txt:
    print("patch_metadata_ext: ANCHOR NOT FOUND — node.py changed upstream;"
          " update this patch", flush=True)
    sys.exit(1)

patched = txt.replace(ANCHOR, REPLACEMENT, 1)
ast.parse(patched)  # build fails here if the patch produced invalid Python
NODE.write_text(patched)
print("patch_metadata_ext: patched node.py — metadata capture is now non-fatal",
      flush=True)
