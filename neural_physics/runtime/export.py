"""Export a `RolloutResult` as a self-contained interactive HTML viewer.

The page decodes the subspace trajectories to vertex positions in the browser (x = x_mu + U^T z, the
same "GPU decompression" idea as Sec. 6.2 of the paper), plays ground truth / network / alpha-beta
side by side or overlaid, colours the network mesh by per-vertex error, and plots the error over time
and the first three PCA components (cf. Fig. 10). Everything -- data, three.js, UI -- is embedded, so
the file can be opened directly from disk and shared.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .rollout import RolloutResult

_HERE = Path(__file__).resolve().parent
_TEMPLATE = _HERE / "viewer.html"
_THREE = _HERE / "vendor" / "three.min.js"
_THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def export_viewer(result: RolloutResult, path, title: str = "Subspace Neural Physics — runtime view",
                  subtitle: str = "", units: str = "", external_label: str = "external state", notes: str = "",
                  include_full_ground_truth: Optional[bool] = None, max_frames: Optional[int] = None,
                  frame_stride: int = 1, inline_three: bool = True) -> Path:
    """Write the viewer to `path` and return it.
    @param include_full_ground_truth: embed the raw ground-truth positions (else the PCA reconstruction is
           shown as ground truth). Default: yes if that costs less than ~25 MB.
    @param max_frames / frame_stride: limit / thin the embedded frames to keep the file small
    @param inline_three: embed the vendored three.js (works offline) instead of loading it from cdnjs
    """
    n = result.z_gt.shape[0]
    sel = np.arange(0, n if max_frames is None else min(n, max_frames * frame_stride), frame_stride)
    if len(sel) < 2:
        raise ValueError("need at least two frames to export")
    x_gt_bytes = len(sel) * result.x_gt.shape[1] * 3 * 4
    if include_full_ground_truth is None:
        include_full_ground_truth = x_gt_bytes <= 25 * 2 ** 20

    e_pred, e_base, e_recon = result.frame_error("pred")[sel], result.frame_error("base")[sel], result.frame_error("recon")[sel]
    v_err = result.vertex_error("pred")[sel]
    scale = float(np.percentile(v_err[np.isfinite(v_err)], 95)) if np.isfinite(v_err).any() else 1.0
    payload = {
        "n_frames": int(len(sel)), "n_verts": int(result.x_gt.shape[1]), "n_components": int(result.z_gt.shape[1]),
        "n_external": int(result.external.shape[1]) if result.external is not None else 0,
        "dt": float(result.dt * frame_stride),
        "mean": _b64(result.pca_mean, np.float32), "basis": _b64(result.pca_basis, np.float32),
        "z_gt": _b64(result.z_gt[sel], np.float32), "z_pred": _b64(np.nan_to_num(result.z_pred[sel], nan=0.0, posinf=1e30, neginf=-1e30), np.float32),
        "z_base": _b64(np.nan_to_num(result.z_base[sel], nan=0.0, posinf=1e30, neginf=-1e30), np.float32),
        "x_gt": _b64(result.x_gt[sel].reshape(len(sel), -1), np.float32) if include_full_ground_truth else None,
        "faces": _b64(result.faces, np.int32) if result.faces is not None else None,
        "external": _b64(result.external[sel], np.float32) if result.external is not None else None,
        "err_pred": _b64(np.nan_to_num(e_pred, nan=1e30), np.float32), "err_base": _b64(np.nan_to_num(e_base, nan=1e30), np.float32),
        "err_recon": _b64(e_recon, np.float32),
        "vertex_error_scale": max(scale, 1e-9),
        "summary": result.summary,
        "meta": {"subtitle": subtitle, "units": units, "external_label": external_label, "notes": notes},
    }
    data_json = json.dumps(payload).replace("</", "<\\/")
    if inline_three and _THREE.exists():
        three_tag = "<script>\n" + _THREE.read_text(encoding="utf-8") + "\n</script>"
    else:
        three_tag = f'<script src="{_THREE_CDN}"></script>'
    html = _TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__NP_TITLE__", _escape(title)).replace("__NP_THREE__", three_tag).replace("__NP_DATA_JSON__", data_json)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
