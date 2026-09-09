import json

import numpy as np
import torch

from neural_physics.core_math.pca import PCA
from neural_physics.runtime.export import export_viewer
from neural_physics.runtime.rollout import evaluate_sequence, simulate
from neural_physics.train.subspace_neural_physics import SubspaceNeuralPhysics


def _toy_positions(n=200, c=10, seed=0):
    """Smoothly moving point cloud (n, c, 3) plus a 2-d external signal."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6, n)[:, None, None]
    base = rng.random((1, c, 3))
    pos = base + 0.1 * np.sin(t * (1 + np.arange(c))[None, :, None] * 0.5) * np.array([1.0, 0.5, 0.2])
    ext = np.stack([np.sin(t[:, 0, 0]), np.cos(t[:, 0, 0])], 1)
    return pos.astype(np.float32), ext.astype(np.float32)


def test_simulate_and_evaluate_sequence(tmp_path):
    pos, ext = _toy_positions()
    X = pos.reshape(pos.shape[0], -1).T
    pca = PCA(4).fit(X)
    Z = pca.encode(X)
    model = SubspaceNeuralPhysics(4, 2, n_layers=3, dt=1 / 30).fit_linear(Z, ext.T)
    z = simulate(model, Z[:, 0], Z[:, 1], ext, clip=True)
    assert z.shape == (200, 4) and np.isfinite(z).all()
    res = evaluate_sequence(model, pca, pos, ext, 1 / 30, faces=np.array([[0, 1, 2]]), start=20, n_frames=100)
    assert res.z_pred.shape == (100, 4) and res.x_pred.shape == (100, 10, 3) and res.x_gt.shape == (100, 10, 3)
    assert res.frame_error("pred").shape == (100,) and res.vertex_error("base").shape == (100, 10)
    s = res.summary
    assert s["n_frames"] == 100 and s["finite"] and s["clipped"]
    assert s["mean_vertex_error"]["pca"] < 1e-3 * 1 + 1.0          # 4 of 4 dominant modes: tiny reconstruction error
    assert set(s["mean_vertex_error"]) == {"network", "alpha_beta", "pca"}
    # roll-out was started from the ground truth frames
    assert np.allclose(res.z_pred[0], res.z_gt[0]) and np.allclose(res.z_pred[1], res.z_gt[1])

    out = export_viewer(res, tmp_path / "viewer.html", title="toy <viewer>", units="m", max_frames=50, inline_three=False)
    html = out.read_text(encoding="utf-8")
    assert "toy &lt;viewer&gt;" in html and "cdnjs.cloudflare.com" in html
    payload = json.loads(html.split('<script id="np-data" type="application/json">')[1].split("</script>")[0].replace("<\\/", "</"))
    assert payload["n_frames"] == 50 and payload["n_verts"] == 10 and payload["n_components"] == 4 and payload["n_external"] == 2
    assert payload["x_gt"] is not None and payload["faces"] is not None
    assert payload["summary"]["n_frames"] == 100
    out2 = export_viewer(res, tmp_path / "viewer2.html", include_full_ground_truth=False, frame_stride=2)
    payload2 = json.loads(out2.read_text(encoding="utf-8").split('<script id="np-data" type="application/json">')[1].split("</script>")[0])
    assert payload2["x_gt"] is None and payload2["n_frames"] == 50 and abs(payload2["dt"] - 2 / 30) < 1e-9
    assert "THREE" in out2.read_text(encoding="utf-8")            # vendored three.js inlined


def test_model_without_external_state():
    pos, _ = _toy_positions()
    X = pos.reshape(pos.shape[0], -1).T
    pca = PCA(3).fit(X)
    Z = pca.encode(X)
    model = SubspaceNeuralPhysics(3, 0, n_layers=3).fit_linear(Z)
    z = simulate(model, Z[:, 0], Z[:, 1], None, n_steps=50)
    assert z.shape == (50, 3)
    zw = torch.tensor(Z[:, :16].T, dtype=torch.float32)[None]
    assert model.rollout(zw[:, 0], zw[:, 1], None, n_steps=16).shape == (1, 16, 3)


def test_pages_snapshot_and_build(tmp_path):
    """End-to-end: save a tiny dataset + model like a training run would, snapshot it, build the site."""
    import yaml
    from neural_physics.data.dataset import SimulationDataset
    from neural_physics.pages import build, snapshot

    pos, ext = _toy_positions(n=240)
    root = tmp_path
    (root / "bin").mkdir()
    (root / "configs").mkdir()
    SimulationDataset(pos, ext, 1 / 30, faces=np.array([[0, 1, 2], [3, 4, 5]]), segments=[(0, 120), (120, 240)],
                      meta={"units": "m"}).save(root / "bin" / "toy.npz")
    cfg = {"name": "toy", "output_dir": "runs/toy", "data": {"dataset": "bin/toy.npz", "test_fraction": 0.5},
           "pca": {"n_components": 4}, "runtime": {"eval_frames": 60, "clip": True, "units": "m"}}
    (root / "configs" / "toy.yml").write_text(yaml.safe_dump(cfg))
    # a "finished run": model.pt + pca.npz in runs/toy
    train_ds, _ = SimulationDataset.load(root / "bin" / "toy.npz").split_segments(0.5)
    pca = PCA(4).fit(train_ds.X())
    model = SubspaceNeuralPhysics(4, 2, n_layers=3, dt=1 / 30).fit_linear(pca.encode(train_ds.X()), train_ds.Y(), train_ds.segments)
    (root / "runs" / "toy").mkdir(parents=True)
    pca.save(root / "runs" / "toy" / "pca.npz")
    model.save(root / "runs" / "toy" / "model.pt")
    snapshot(root / "runs" / "toy", root / "models" / "toy")
    assert (root / "models" / "toy" / "model.pt").exists() and (root / "models" / "toy" / "pca.npz").exists()

    pages = {"title": "test site", "repo": "https://example.org/repo",
             "models": [{"name": "toy", "title": "Toy <model>", "config": "configs/toy.yml", "model_dir": "models/toy",
                         "viewer": {"max_frames": 30, "full_ground_truth": False}}]}
    (root / "configs" / "pages.yml").write_text(yaml.safe_dump(pages))
    site = build(root / "configs" / "pages.yml", root / "site", root=root)
    index = (site / "index.html").read_text(encoding="utf-8")
    assert (site / "toy" / "index.html").exists() and (site / ".nojekyll").exists()
    assert "Toy &lt;model&gt;" in index and "toy/index.html" in index and "https://example.org/repo" in index
    viewer = (site / "toy" / "index.html").read_text(encoding="utf-8")
    assert '"n_frames": 30' in viewer and '"x_gt": null' in viewer
