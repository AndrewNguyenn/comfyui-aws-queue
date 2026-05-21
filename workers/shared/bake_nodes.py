"""
Build-time custom-node baker.

Run during `docker build` (see workers/image/Dockerfile). Clones + installs
every pack listed in baked_nodes.txt into /opt/comfy/custom_nodes/, reusing
manifest_installer's install logic (numpy/torch pin constraint, single clean
OpenCV, cv2 typing patch). Drops a `.baked` marker in each pack dir so the
runtime manifest_installer skips it — moving the ~13 min of node installs off
the worker cold-start path and into the image.

A pack that fails to install is logged and skipped — it doesn't fail the whole
build (and the build's smoke test still catches a broken ComfyUI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import manifest_installer as mi


def main(list_path: str) -> int:
    entries = []
    for raw in Path(list_path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        entries.append((parts[0], parts[1], parts[2] if len(parts) > 2 else None))

    print(f"[bake] installing {len(entries)} custom-node packs", flush=True)
    mi.CUSTOM_NODES_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for name, url, commit in entries:
        try:
            mi._install_one(name, url, commit)
            (mi.CUSTOM_NODES_DIR / name / ".baked").write_text("baked at image build\n")
            print(f"[bake] OK   {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"[bake] FAIL {name}: {e!r}", flush=True)

    # Repair the foundation after the whole set is installed: one clean
    # OpenCV (mixed variants crash ComfyUI) + transformers pinned to a
    # torch-2.5-compatible release (newer ones break every transformers
    # node). See manifest_installer for the details.
    mi._force_clean_opencv()
    mi._force_transformers()
    mi._restore_torchaudio_stub()

    print(f"[bake] done — {len(entries) - len(failed)} ok, {len(failed)} failed: {failed}",
          flush=True)
    # Non-fatal: a failed pack just won't load. The image still builds.
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/baked_nodes.txt"))
