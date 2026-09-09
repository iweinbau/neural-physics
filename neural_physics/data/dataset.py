"""Container for a simulation time series in the format used throughout the package.

Paper (Sec. 4 / 5.1): the only input is a raw time series of vertex positions of the simulated
object, x_t in R^{3c}, and a per-frame vector y_t in R^e describing the external objects
(ball position, wind speed and direction, joint positions, ...).

`SimulationDataset` stores exactly that plus what is needed to *evaluate and render* a model:
mesh faces (optional), the boundaries of the individual simulation episodes (so training windows
never straddle a reset) and the frame time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SimulationDataset:
    positions: np.ndarray                 # (n_frames, n_verts, 3)
    external: np.ndarray                  # (n_frames, e)
    dt: float
    faces: Optional[np.ndarray] = None    # (n_faces, 3) int, optional (for rendering)
    segments: Optional[np.ndarray] = None  # (n_segments, 2) [start, end) frame indices
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=np.float32)
        if self.positions.ndim == 2:  # (n_frames, 3c) -> (n_frames, c, 3)
            self.positions = self.positions.reshape(self.positions.shape[0], -1, 3)
        assert self.positions.ndim == 3 and self.positions.shape[2] == 3, "positions must be (n, c, 3) or (n, 3c)"
        self.external = np.asarray(self.external, dtype=np.float32)
        if self.external.ndim == 1:
            self.external = self.external[:, None]
        assert self.external.shape[0] == self.positions.shape[0], "positions and external must have the same number of frames"
        if self.faces is not None:
            self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.segments is None:
            self.segments = np.array([[0, self.n_frames]], dtype=np.int64)
        self.segments = np.asarray(self.segments, dtype=np.int64).reshape(-1, 2)
        assert self.segments[:, 0].min() >= 0 and self.segments[:, 1].max() <= self.n_frames

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_verts(self) -> int:
        return self.positions.shape[1]

    @property
    def n_external(self) -> int:
        return self.external.shape[1]

    def X(self) -> np.ndarray:
        """Positions as the paper's X matrix: (3c, n_frames)."""
        return self.positions.reshape(self.n_frames, -1).T

    def Y(self) -> np.ndarray:
        """External state as the paper's Y matrix: (e, n_frames)."""
        return self.external.T

    def split_segments(self, test_fraction: float = 0.2, seed: int = 0) -> Tuple["SimulationDataset", "SimulationDataset"]:
        """Split into train / test *by episode* so the test set is truly unseen.

        With a single episode the last `test_fraction` of the frames is held out instead.
        """
        if len(self.segments) == 1:
            s, e = self.segments[0]
            cut = int(s + (e - s) * (1.0 - test_fraction))
            return self.subset([(s, cut)]), self.subset([(cut, e)])
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.segments))
        n_test = max(1, int(round(len(self.segments) * test_fraction)))
        test_idx, train_idx = sorted(idx[:n_test]), sorted(idx[n_test:])
        return self.subset([tuple(self.segments[i]) for i in train_idx]), self.subset([tuple(self.segments[i]) for i in test_idx])

    def subset(self, ranges: Sequence[Tuple[int, int]]) -> "SimulationDataset":
        """Concatenate the given [start, end) frame ranges into a new dataset (each range = one segment)."""
        pos, ext, segs, cursor = [], [], [], 0
        for s, e in ranges:
            pos.append(self.positions[s:e]); ext.append(self.external[s:e])
            segs.append((cursor, cursor + (e - s))); cursor += e - s
        return SimulationDataset(np.concatenate(pos), np.concatenate(ext), self.dt, self.faces, np.array(segs), dict(self.meta))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            positions=self.positions,
            external=self.external,
            dt=np.float64(self.dt),
            faces=self.faces if self.faces is not None else np.zeros((0, 3), dtype=np.int64),
            segments=self.segments,
            meta=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path) -> "SimulationDataset":
        with np.load(path, allow_pickle=False) as f:
            faces = f["faces"] if f["faces"].size else None
            meta = json.loads(str(f["meta"])) if "meta" in f else {}
            return cls(f["positions"], f["external"], float(f["dt"]), faces, f["segments"], meta)

    @classmethod
    def from_legacy_npy(cls, data_file, external_file, dt: float, faces_file=None) -> "SimulationDataset":
        """Load the two-`.npy` layout written by `bin/datagen` (frames x 3c and frames x e)."""
        faces = None
        if faces_file is not None:
            faces = load_faces(faces_file)
        return cls(np.load(data_file), np.load(external_file), dt, faces)


def load_faces(path) -> np.ndarray:
    """Read triangle faces from a `.npy` file or the `f` lines of a Wavefront `.obj` (quads are split)."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.int64)
    faces: List[List[int]] = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("f "):
                continue
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(faces, dtype=np.int64)
