"""Build-time patch — make torchaudio's C++ extension load non-fatal.

MiniMax H3 is native in ComfyUI and comfy_extras/nodes_minimax_h3.py does a bare
`import torchaudio` at module load, so without a working import EVERY MiniMax
node silently disappears from the graph.

We cannot use the torchaudio that ComfyUI's requirements resolve to (2.11, built
for torch 2.11 / CUDA 13) against this NGC base (2.5.0a0+...nv24.10, CUDA 12.6)
— that is the libcudart.so.13 mismatch which crash-loops ComfyUI. But the
matching 2.5.1 wheel does not import either: its prebuilt libtorchaudio.so is
linked against upstream torch 2.5.1, while NGC ships a custom build, so loading
it dies with

    OSError: libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv

That failure is fatal because _extension/__init__.py loads the library at import
with no guard ("we do not catch the failure as it suggests there is something
wrong with the installation").

Here the installation is fine — only the compiled half is unusable, and we do
not need it. Everything ComfyUI actually calls (functional.resample,
transforms.MelSpectrogram / MelScale, used by MiniMax H3, ACE, MMAudio,
Lightricks and gemma4) is pure PyTorch DSP. The C++ ops cover file I/O, sox
effects, RIR and forced alignment, none of which we use — and RIR/align already
degrade through fail_with_message when unavailable.

So: wrap the extension load, and on failure fall back to the same state
torchaudio uses for a build compiled without it. Exits non-zero (fails the
build) if the anchor is missing, so a torchaudio upgrade that reshapes this file
is caught at build time rather than by every MiniMax node vanishing later.
"""

import ast
import pathlib
import sys

TARGET = pathlib.Path(
    "/usr/local/lib/python3.10/dist-packages/torchaudio/_extension/__init__.py"
)

ANCHOR = """if _IS_TORCHAUDIO_EXT_AVAILABLE:
    _load_lib("libtorchaudio")

    import torchaudio.lib._torchaudio  # noqa

    _check_cuda_version()
    _IS_RIR_AVAILABLE = torchaudio.lib._torchaudio.is_rir_available()
    _IS_ALIGN_AVAILABLE = torchaudio.lib._torchaudio.is_align_available()"""

REPLACEMENT = """if _IS_TORCHAUDIO_EXT_AVAILABLE:
    # PATCHED (workers/video/patch_torchaudio_ext.py): the prebuilt
    # libtorchaudio.so is linked against upstream torch, not NGC's custom
    # build, so loading it raises an undefined-symbol OSError. The pure-Python
    # DSP (functional.resample, transforms.MelSpectrogram) is all we use, so
    # degrade to the "compiled without the extension" state instead of dying
    # and taking every MiniMax H3 node with us.
    try:
        _load_lib("libtorchaudio")

        import torchaudio.lib._torchaudio  # noqa

        _check_cuda_version()
        _IS_RIR_AVAILABLE = torchaudio.lib._torchaudio.is_rir_available()
        _IS_ALIGN_AVAILABLE = torchaudio.lib._torchaudio.is_align_available()
    except Exception as _e:  # noqa: BLE001
        _LG.warning(
            "torchaudio C++ extension unavailable (%s); "
            "pure-python DSP still works, file I/O and sox do not.", _e
        )
        _IS_TORCHAUDIO_EXT_AVAILABLE = False"""


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found", file=sys.stderr)
        return 1
    src = TARGET.read_text()
    if "PATCHED (workers/video/patch_torchaudio_ext.py)" in src:
        print("torchaudio extension already patched")
        return 0
    if ANCHOR not in src:
        print(f"ERROR: anchor not found in {TARGET} — torchaudio layout changed",
              file=sys.stderr)
        return 1
    patched = src.replace(ANCHOR, REPLACEMENT, 1)
    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"ERROR: patched file does not parse: {e}", file=sys.stderr)
        return 1
    TARGET.write_text(patched)
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
