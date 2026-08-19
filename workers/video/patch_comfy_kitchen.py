"""Build-time patch — make comfy_kitchen's na3d op registerable on torch 2.5.

ComfyUI >= v0.30 (which is where native MiniMax H3 lives) imports comfy_kitchen
from comfy/quant_ops.py, and comfy_kitchen registers a custom op whose signature
uses PEP 585 builtin generics:

    @torch.library.custom_op("comfy_kitchen::na3d", mutates_args=())
    def _op_na3d(..., kernel_size: list[int], is_causal: list[bool],
                 scale: float | None) -> torch.Tensor

torch 2.5 (what this NGC base ships) parses custom-op signatures against an
explicit table of accepted annotations. That table contains typing.List[int] and
typing.Optional[float] but NOT the lowercase list[int] / float | None spellings,
so registration raises at import — which propagates all the way out of
`import comfy.utils` and crash-loops ComfyUI before it ever serves a job:

    ValueError: unsupported type annotation list[int]. The valid types are:
    dict_keys([... typing.List[int] ... typing.Optional[float] ...])

The fix is a spelling change, not a feature removal: rewrite that one signature
to the typing.* forms torch 2.5 accepts. na3d (3D neighborhood attention) keeps
working, which matters because disabling it would silently break any model that
does use it — MiniMax H3 does not, but other models in the same ComfyUI do.

Anchor-based and idempotent; exits non-zero if the signature moves, so a
comfy_kitchen upgrade that reshapes this file fails the BUILD rather than
crash-looping every worker.
"""

import ast
import pathlib
import sys

TARGET = pathlib.Path(
    "/usr/local/lib/python3.10/dist-packages/comfy_kitchen/backends/eager/na.py"
)

ANCHOR = """@torch.library.custom_op("comfy_kitchen::na3d", mutates_args=())
def _op_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
    is_causal: list[bool],
    scale: float | None,
) -> torch.Tensor:"""

REPLACEMENT = """# PATCHED (workers/video/patch_comfy_kitchen.py): torch 2.5 accepts
# typing.List[...] / typing.Optional[...] in a custom-op signature but not the
# PEP 585 list[...] / X | None spellings. Same types, spelling torch can parse.
import typing as _t  # noqa: E402


@torch.library.custom_op("comfy_kitchen::na3d", mutates_args=())
def _op_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: _t.List[int],
    is_causal: _t.List[bool],
    scale: _t.Optional[float],
) -> torch.Tensor:"""


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found", file=sys.stderr)
        return 1
    src = TARGET.read_text()
    if "PATCHED (workers/video/patch_comfy_kitchen.py)" in src:
        print("comfy_kitchen na3d already patched")
        return 0
    if ANCHOR not in src:
        print(f"ERROR: na3d custom-op signature not found in {TARGET} — "
              "comfy_kitchen layout changed", file=sys.stderr)
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
