"""Export a trained checkpoint to ONNX for browser (and Node) inference.

Writes `public/models/set_id_<arch>.onnx` plus a sibling `.json` carrying
img_size and arch, since onnxruntime-web doesn't surface model metadata.
The exporter emits four named outputs (number_logits, color_logits,
shape_logits, shading_logits) so the JS side can pluck them directly without
depending on head order.

Post-export, the graph is patched down to opset 13 for WebGL compatibility.
The dynamo exporter targets opset 18+ which uses forms that onnxruntime-web's
WebGL backend chokes on:
  - ReduceMean passes axes as a second input tensor (opset 18+) instead of
    an attribute (opset 13). WebGL errors with "Reduce op requires 1 input".
  - Reshape emits an `allowzero` attribute (opset 14+) that WebGL doesn't
    recognize.
The patch moves ReduceMean axes back to an attribute, strips allowzero from
Reshape, and stamps the graph as opset 13.

Usage:
    uv run python -m set_id.export_onnx                      # smaller_best.pt → public/models/set_id_smaller.onnx
    uv run python -m set_id.export_onnx --ckpt checkpoints/small_best.pt
    uv run python -m set_id.export_onnx --out /tmp/foo.onnx
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

# Pre-3.14 pickles reference `pathlib._local.PosixPath`; keep old ckpts loadable.
sys.modules.setdefault("pathlib._local", pathlib)

import torch

from set_id.model import build_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = REPO_ROOT / "ml" / "checkpoints" / "smaller_best.pt"

WEBGL_TARGET_OPSET = 13


def _patch_for_webgl(model_path: Path) -> int:
    """Downgrade an exported ONNX graph to opset 13 for WebGL compatibility.

    Returns the target opset version stamped into the model.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(model_path))
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    for node in model.graph.node:
        if node.op_type == "ReduceMean" and len(node.input) > 1:
            axes_val = inits[node.input[1]]
            del node.input[1]
            while len(node.attribute):
                node.attribute.pop()
            node.attribute.append(helper.make_attribute("axes", axes_val.tolist()))
            node.attribute.append(helper.make_attribute("keepdims", 1))

        if node.op_type == "Reshape":
            to_remove = [a for a in node.attribute if a.name == "allowzero"]
            for a in to_remove:
                node.attribute.remove(a)

    for oi in model.opset_import:
        if oi.domain in ("", "ai.onnx"):
            oi.version = WEBGL_TARGET_OPSET

    onnx.checker.check_model(model)
    onnx.save(model, str(model_path))
    return WEBGL_TARGET_OPSET


def export(ckpt_path: Path, out_path: Path, opset: int | None = None) -> dict:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("cfg") or {}
    arch = cfg.get("arch") or "smaller"
    img_size = int(cfg.get("img_size") or 128)

    model = build_model(arch, pretrained=False)
    model.load_state_dict(state["state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, img_size, img_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Head order inside the module is (number, color, shape, shading) — see
    # `set_id.model._Heads`. We expose those as named outputs so the JS side
    # never has to care about positional order.
    # Uses the torch 2.9+ dynamo-based exporter (the legacy TorchScript path
    # is deprecated). `dynamic_shapes` is the dynamo equivalent of the old
    # `dynamic_axes`; `Dim("batch")` makes dim 0 dynamic. `external_data=False`
    # keeps the weights inline in the .onnx file (default writes them to a
    # sibling .onnx.data, which is awkward for browser deployment).
    export_kwargs: dict = {
        "input_names": ["pixels"],
        "output_names": [
            "number_logits",
            "color_logits",
            "shape_logits",
            "shading_logits",
        ],
        "external_data": False,
    }
    if opset is not None:
        export_kwargs["opset_version"] = opset
    torch.onnx.export(model, (dummy,), str(out_path), **export_kwargs)

    # Round-trip with onnxruntime + compare against torch to catch export drift.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        sample = torch.randn(1, 3, img_size, img_size)
        with torch.inference_mode():
            torch_out = model(sample)
        ort_out = sess.run(None, {"pixels": sample.numpy()})
        max_abs = max(
            float(np.abs(t.numpy() - o).max()) for t, o in zip(torch_out, ort_out)
        )
        print(f"ONNX parity check: max |torch - ort| = {max_abs:.2e}")
    except ImportError:
        print("(skipping parity check — onnxruntime not installed)")

    actual_opset = _patch_for_webgl(out_path)
    print(f"patched graph to opset {actual_opset} for WebGL compatibility")

    return {"arch": arch, "img_size": img_size, "opset": actual_opset}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .onnx path. Defaults to public/models/set_id_<arch>.onnx",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Override opset version. Default: let the dynamo exporter pick.",
    )
    args = p.parse_args()

    ckpt_path = args.ckpt
    if not ckpt_path.is_absolute():
        # Resolve relative to ml/ so `--ckpt checkpoints/x.pt` works from ml/.
        ckpt_path = (REPO_ROOT / "ml" / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    # Peek at cfg to name the default output file.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = (state.get("cfg") or {}).get("arch") or "smaller"

    out_path = args.out or (REPO_ROOT / "public" / "models" / f"set_id_{arch}.onnx")
    # Clean up any sibling external-data file left by an earlier export run.
    ext_data = out_path.with_suffix(out_path.suffix + ".data")
    if ext_data.exists():
        ext_data.unlink()
    info = export(ckpt_path, out_path, opset=args.opset)
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(info, indent=2) + "\n")
    size_kb = out_path.stat().st_size / 1024
    opset_str = info["opset"] if info["opset"] is not None else "default"
    print(
        f"wrote {out_path} ({size_kb:,.1f} KB) — arch={info['arch']} img_size={info['img_size']} opset={opset_str}"
    )
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
