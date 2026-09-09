import numpy as np

from neural_physics.data.cloth_sim import ClothConfig, ClothSim, generate
from neural_physics.data.dataset import SimulationDataset, load_faces


def test_dataset_roundtrip_and_split(tmp_path):
    rng = np.random.default_rng(0)
    ds = SimulationDataset(rng.random((120, 7, 3)), rng.random((120, 2)), 1 / 60, faces=np.array([[0, 1, 2], [2, 3, 4]]),
                           segments=[(0, 50), (50, 120)], meta={"units": "m"})
    assert ds.X().shape == (21, 120) and ds.Y().shape == (2, 120)
    p = tmp_path / "ds.npz"
    ds.save(p)
    back = SimulationDataset.load(p)
    assert np.allclose(back.positions, ds.positions) and np.allclose(back.external, ds.external)
    assert back.dt == ds.dt and back.meta == {"units": "m"} and np.array_equal(back.faces, ds.faces)
    train, test = ds.split_segments(0.5, seed=0)
    assert train.n_frames + test.n_frames == 120 and len(train.segments) == 1 and len(test.segments) == 1
    # flattened (frames x 3c) input is accepted
    flat = SimulationDataset(rng.random((10, 21)), rng.random((10, 1)), 0.1)
    assert flat.positions.shape == (10, 7, 3) and flat.segments.tolist() == [[0, 10]]
    single_train, single_test = flat.split_segments(0.2)
    assert single_train.n_frames == 8 and single_test.n_frames == 2


def test_legacy_npy_and_obj_faces(tmp_path):
    rng = np.random.default_rng(1)
    np.save(tmp_path / "data.npy", rng.random((30, 12)))
    np.save(tmp_path / "external.npy", rng.random((30, 3)))
    (tmp_path / "mesh.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    faces = load_faces(tmp_path / "mesh.obj")
    assert faces.tolist() == [[0, 1, 2], [0, 2, 3]]                # quad split into two triangles, 0-based
    ds = SimulationDataset.from_legacy_npy(tmp_path / "data.npy", tmp_path / "external.npy", 1 / 60, tmp_path / "mesh.obj")
    assert ds.n_verts == 4 and ds.n_external == 3 and ds.faces.shape == (2, 3)


def test_cloth_sim_small():
    cfg = ClothConfig(nx=6, ny=4, substeps=2, iterations=4)
    sim = ClothSim(cfg)
    for i, j, rest, si, sj in sim.groups:                          # constraint groups are vertex-disjoint
        allv = np.concatenate([i, j])
        assert len(np.unique(allv)) == len(allv)
    ds = generate(episodes=2, frames_per_episode=15, cfg=cfg, seed=0, verbose=False)
    assert ds.positions.shape == (30, 24, 3) and ds.external.shape == (30, 3)
    assert np.isfinite(ds.positions).all() and ds.faces.shape == (2 * 5 * 3, 3)
    assert ds.segments.tolist() == [[0, 15], [15, 30]]
    pinned = ds.positions[:, ::cfg.nx]                             # column 0 stays on the pole
    assert np.allclose(pinned, pinned[0])
    free = ds.positions[:, 1:]
    assert not np.allclose(free[0], free[-1])                      # the rest of the cloth moves
