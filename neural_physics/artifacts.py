"""Loading a trained run's artifacts (model + PCA basis) with consistency checks.

A run directory (`runs/<name>/`, or a published snapshot in `models/<name>/`) holds two files that
only make sense together: `model.pt` (alpha/beta, normalisation, clipping ranges and the network,
all sized by the number of PCA components) and `pca.npz` (the basis those components refer to).
Nothing in the maths ties them together, so a leftover file from an earlier run with a different
`pca.n_components` used to fail deep inside the roll-out with an opaque tensor-size error. The
helpers here catch that up front and say what to do about it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from neural_physics.core_math.pca import PCA
from neural_physics.train.subspace_neural_physics import SubspaceNeuralPhysics


class ArtifactMismatch(RuntimeError):
    """Model and PCA basis (or dataset) do not belong together."""


def basis_fingerprint(pca: PCA) -> str:
    """Short stable id of a basis: `<components>x<dof>:<hash>`. Detects a re-fitted basis of the
    same size (different split, different subsample) as well as a different size."""
    digest = hashlib.sha1(np.ascontiguousarray(pca.U, dtype=np.float64).tobytes()).hexdigest()[:12]
    return f"{pca.n_components}x{pca.U.shape[1]}:{digest}"


def _age_hint(model_path: Optional[Path], pca_path: Optional[Path]) -> str:
    if not (model_path and pca_path and Path(model_path).exists() and Path(pca_path).exists()):
        return ""
    dt = Path(pca_path).stat().st_mtime - Path(model_path).stat().st_mtime
    if dt < -60:
        return (f"\n  {Path(pca_path).name} is {abs(dt) / 60:.0f} min older than {Path(model_path).name}, so it is "
                f"probably left over from an earlier run (training only wrote the basis at the very end before "
                f"neural-physics 0.1.1).")
    if dt > 60:
        return f"\n  {Path(pca_path).name} is newer than {Path(model_path).name} -- the basis was re-fitted after training."
    return ""


def check_compatible(model: SubspaceNeuralPhysics, pca: PCA, n_verts: Optional[int] = None,
                     n_external: Optional[int] = None, model_path=None, pca_path=None) -> None:
    """Raise `ArtifactMismatch` when the pieces cannot be used together. Cheap; call it before any roll-out."""
    if pca.n_components != model.n_components:
        raise ArtifactMismatch(
            f"the model was trained on {model.n_components} PCA components but the basis has {pca.n_components}."
            f"\n  model: {model_path or '<model>'}\n  basis: {pca_path or '<pca.npz>'}"
            f"{_age_hint(model_path, pca_path)}"
            f"\n  Fix: re-run training so both are written together (`pdm run train <config>`), or point the config's "
            f"`output_dir` / `model_dir` at the run that produced this model.")
    if n_verts is not None and pca.U.shape[1] != 3 * n_verts:
        raise ArtifactMismatch(
            f"the basis maps {pca.U.shape[1]} degrees of freedom but the dataset has {n_verts} vertices "
            f"({3 * n_verts} DOF) -- basis and dataset do not match."
            f"\n  basis: {pca_path or '<pca.npz>'}"
            f"\n  Fix: use the dataset this model was trained on, or retrain on the current one.")
    if n_external is not None and model.n_external != n_external:
        raise ArtifactMismatch(
            f"the model expects {model.n_external} external-state dimensions but the data provides {n_external}."
            f"\n  Fix: check `pca.external_components` in the config, or retrain.")


def divergence_warnings(model: SubspaceNeuralPhysics, pca: PCA, config: Optional[Dict] = None,
                        expect_n_components: Optional[int] = None) -> List[str]:
    """Non-fatal differences worth telling the user about: a re-fitted basis, or a config that has been
    edited since the model was trained (a changed split makes the "held-out" episode meaningless)."""
    out: List[str] = []
    meta = getattr(model, "meta", None) or {}
    if meta.get("basis") and meta["basis"] != basis_fingerprint(pca):
        out.append(f"the basis does not fingerprint as the one used for training "
                   f"({basis_fingerprint(pca)} vs {meta['basis']}) -- same size, different vectors")
    if expect_n_components is not None and expect_n_components != pca.n_components:
        out.append(f"the config asks for {expect_n_components} PCA components but these artifacts have "
                   f"{pca.n_components}; the config change only takes effect after retraining")
    trained_on = meta.get("data") or {}
    for key in ("dataset", "test_fraction", "split_seed"):
        now = (config or {}).get("data", {}).get(key)
        then = trained_on.get(key)
        if then is not None and now is not None and str(then) != str(now):
            out.append(f"data.{key} is now {now!r} but the model was trained with {then!r} -- the evaluation "
                       f"episode may not be held out for this model")
    return out


def load_run(run_dir, model_file: str = "model.pt", n_verts: Optional[int] = None,
             n_external: Optional[int] = None) -> Tuple[SubspaceNeuralPhysics, PCA, Optional[PCA]]:
    """Load `model.pt` + `pca.npz` (+ `pca_external.npz`) from a run directory or snapshot, checked."""
    run_dir = Path(run_dir)
    model_path, pca_path = run_dir / model_file, run_dir / "pca.npz"
    missing = [f"{what} ({path})" for path, what in ((model_path, "trained model"), (pca_path, "PCA basis"))
               if not path.exists()]
    if missing:
        raise ArtifactMismatch("not found: " + ", ".join(missing)
                               + f"\n  Fix: run `pdm run train <config>` first -- it writes the model and its basis "
                                 f"into {run_dir} (and `pdm run publish-model <run> <model_dir>` copies them for the site).")
    model = SubspaceNeuralPhysics.load(model_path)
    pca = PCA.load(pca_path)
    ext_pca = PCA.load(run_dir / "pca_external.npz") if (run_dir / "pca_external.npz").exists() else None
    check_compatible(model, pca, n_verts, n_external, model_path=model_path, pca_path=pca_path)
    return model, pca, ext_pca
