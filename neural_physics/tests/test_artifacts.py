import numpy as np
import pytest
import torch
import yaml

from neural_physics.artifacts import ArtifactMismatch, basis_fingerprint, check_compatible, divergence_warnings, load_run
from neural_physics.core_math.pca import PCA
from neural_physics.data.dataset import SimulationDataset
from neural_physics.runtime.export import export_viewer
from neural_physics.runtime.rollout import evaluate_sequence
from neural_physics.train.subspace_neural_physics import SubspaceNeuralPhysics


def _toy(n=240, c=12, seed=0):
    """Smoothly moving point cloud (n, c, 3) plus a 2-d external signal."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6, n)[:, None, None]
    pos = rng.random((1, c, 3)) + 0.1 * np.sin(t * (1 + np.arange(c))[None, :, None] * 0.5) * np.array([1.0, 0.5, 0.2])
    ext = np.stack([np.sin(t[:, 0, 0]), np.cos(t[:, 0, 0])], 1)
    return pos.astype(np.float32), ext.astype(np.float32)


def _fitted(u, pos, ext, n_layers=3):
    X = pos.reshape(pos.shape[0], -1).T
    pca = PCA(u).fit(X)
    model = SubspaceNeuralPhysics(u, ext.shape[1], n_layers=n_layers, dt=1 / 30).fit_linear(pca.encode(X), ext.T)
    return pca, model


# --------------------------------------------------------------------------- the visualisation is basis-size agnostic
@pytest.mark.parametrize("u", [2, 5, 9, 17])
def test_viewer_export_matches_any_basis_size(u, tmp_path):
    """Regression guard: nothing in the runtime or the viewer payload may assume a fixed number of PCA
    components (the cloth config happens to use 32/64, which must not leak into the visualisation)."""
    pos, ext = _toy()
    pca, model = _fitted(u, pos, ext)
    res = evaluate_sequence(model, pca, pos, ext, 1 / 30, faces=np.array([[0, 1, 2]]), start=0, n_frames=40)
    assert res.z_pred.shape == (40, u) and res.summary["n_components"] == u
    assert res.pca_basis.shape == (u, 3 * pos.shape[1])
    out = export_viewer(res, tmp_path / f"v{u}.html", title=f"u={u}", inline_three=False)
    html = out.read_text(encoding="utf-8")
    assert f'"n_components": {u}' in html
    # the decode loop in the viewer reads basis as (u x 3c) floats: the payload must carry exactly that many
    import base64 as b64
    import json
    payload = json.loads(html.split('<script id="np-data" type="application/json">')[1].split("</script>")[0].replace("<\\/", "</"))
    assert len(b64.b64decode(payload["basis"])) == u * 3 * pos.shape[1] * 4
    assert len(b64.b64decode(payload["mean"])) == 3 * pos.shape[1] * 4


# --------------------------------------------------------------------------- mismatch detection
def test_check_compatible_reports_component_mismatch(tmp_path):
    pos, ext = _toy()
    pca4, _ = _fitted(4, pos, ext)
    _, model6 = _fitted(6, pos, ext)
    with pytest.raises(ArtifactMismatch) as err:
        check_compatible(model6, pca4, model_path=tmp_path / "model.pt", pca_path=tmp_path / "pca.npz")
    msg = str(err.value)
    assert "6 PCA components" in msg and "4" in msg and "pdm run train" in msg      # actionable, names both sizes

    pca_ok, model_ok = _fitted(4, pos, ext)
    check_compatible(model_ok, pca_ok, n_verts=pos.shape[1], n_external=ext.shape[1])   # the matching pair is fine
    with pytest.raises(ArtifactMismatch, match="degrees of freedom"):
        check_compatible(model_ok, pca_ok, n_verts=pos.shape[1] + 1)
    with pytest.raises(ArtifactMismatch, match="external-state"):
        check_compatible(model_ok, pca_ok, n_external=ext.shape[1] + 1)


def test_load_run_rejects_a_stale_basis(tmp_path):
    """The exact trap: a basis left over from an earlier run next to a newer checkpoint."""
    pos, ext = _toy()
    pca_old, _ = _fitted(4, pos, ext)
    _, model_new = _fitted(6, pos, ext)
    pca_old.save(tmp_path / "pca.npz")
    model_new.save(tmp_path / "model.pt")
    with pytest.raises(ArtifactMismatch) as err:
        load_run(tmp_path)
    assert "pca.npz" in str(err.value) and "model.pt" in str(err.value)
    # missing files are reported by name, not as a KeyError/FileNotFoundError deep inside
    with pytest.raises(ArtifactMismatch, match="not found: trained model .*, PCA basis "):
        load_run(tmp_path / "empty")


def test_divergence_warnings_flag_edited_config_and_refit_basis():
    pos, ext = _toy()
    pca, model = _fitted(5, pos, ext)
    model.meta = {"basis": basis_fingerprint(pca), "data": {"dataset": "bin/a.npz", "test_fraction": 0.2, "split_seed": 0}}
    assert divergence_warnings(model, pca, {"data": {"dataset": "bin/a.npz", "test_fraction": 0.2, "split_seed": 0}}) == []
    # config asks for a different basis size than the artifacts have
    warns = divergence_warnings(model, pca, expect_n_components=64)
    assert any("64" in w and "retraining" in w for w in warns)
    # a re-fitted basis of the same size
    other, _ = _fitted(5, *_toy(seed=3))
    assert any("different vectors" in w for w in divergence_warnings(model, other))
    # a changed split
    warns = divergence_warnings(model, pca, {"data": {"dataset": "bin/a.npz", "test_fraction": 0.5, "split_seed": 0}})
    assert any("test_fraction" in w for w in warns)


# --------------------------------------------------------------------------- the pipeline keeps a run consistent
def test_pipeline_writes_basis_before_training_and_records_provenance(tmp_path):
    from neural_physics.pipeline import run

    pos, ext = _toy(n=200)
    (tmp_path / "bin").mkdir()
    SimulationDataset(pos, ext, 1 / 30, segments=[(0, 100), (100, 200)]).save(tmp_path / "bin" / "toy.npz")
    out = tmp_path / "runs" / "toy"
    config = {"name": "toy", "output_dir": str(out), "data": {"dataset": str(tmp_path / "bin" / "toy.npz"), "test_fraction": 0.5},
              "pca": {"n_components": 5}, "model": {"n_layers": 3}, "train": {"epochs": 1, "steps_per_epoch": 2, "eval_windows": 8},
              "runtime": {"eval_frames": 40, "viewer": "viewer.html"}}
    summary = run(config, verbose=False)
    assert summary["runtime"]["n_components"] == 5

    # the basis is on disk and agrees with every checkpoint written during training
    model, pca, _ = load_run(out, model_file="checkpoints/best.pt", n_verts=pos.shape[1])
    assert pca.n_components == 5
    assert (out / "pca.npz").stat().st_mtime <= (out / "checkpoints" / "best.pt").stat().st_mtime
    assert model.meta["basis"] == basis_fingerprint(pca)
    assert model.meta["data"]["test_fraction"] == 0.5

    # editing the config afterwards must not silently re-export the old basis
    config["pca"]["n_components"] = 9
    summary = run(config, eval_only=True, verbose=False)
    assert summary["runtime"]["n_components"] == 5      # the artifacts win, and the warning says so
    assert summary["pca"]["n_components"] == 5
