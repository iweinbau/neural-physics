# neural-physics

An implementation of [Subspace Neural Physics: Fast Data-Driven Interactive Simulation](https://dl.acm.org/doi/10.1145/3309486.3340245)
(Holden, Duong, Datta, Nowrouzezahrai, SCA 2019).

```
positions x_t  --PCA-->  z_t          z_bar_t = alpha ⊙ z_{t-1} + beta ⊙ (z_{t-1} - z_{t-2})      (Eq. 1)
external  y_t  ------->  w_t          z_t     = z_bar_t + Phi([z_bar_t, z_{t-1}, w_t])            (Eq. 3)
```

`Phi` is trained with the paper's roll-out procedure (Algorithm 1): mini-batches of 16 random
32-frame windows, noisy initial states, the network's own predictions fed back in, mean absolute
position + velocity loss, AmsGrad. At runtime the model is rolled out auto-regressively with the
Sec. 8.1 clipping and decoded back to vertices with the PCA basis.

## Install

The project is managed with [PDM](https://pdm-project.org) and needs Python >= 3.12.

```bash
pipx install pdm          # or: brew install pdm / pip install --user pdm
pdm install -G dev        # creates .venv from pdm.lock: numpy, torch, pyyaml (+ pytest, flake8, matplotlib)
pdm run check             # flake8 + pytest (~20 s)
pdm run lint-style        # advisory style report (never fails)
pdm run --list            # all project commands
```

`pdm install -G tensorboard` adds TensorBoard (then set `train.tensorboard: true` in a config). On Linux the
default torch wheels from PyPI include CUDA; uncomment the `pytorch-cpu` source in `pyproject.toml` and
re-lock for CPU-only wheels.

## Quick start: synthetic flag on a pole

```bash
pdm run datagen-cloth -o bin/cloth.npz      # 8 wind episodes x 2500 frames of a 25x17 PBD cloth (~5 min)
pdm run train-cloth                         # PCA -> alpha/beta -> train 100 epochs -> runs/cloth/viewer.html
pdm run train-cloth --max-time 1800         # or stop after 30 minutes
pdm run viewer-cloth                        # re-export the viewer from runs/cloth/checkpoints/best.pt
pdm run train configs/cloth.yml --epochs 5  # any config directly (same as the np-train console script)
```

Open `runs/cloth/viewer.html` in a browser (no server needed): ground truth, network roll-out and
the alpha/beta-only roll-out play side by side or overlaid, the network mesh is coloured by per-vertex
error, and the panel plots the error over time and the first three PCA components (cf. Fig. 10 of the
paper). Two held-out wind episodes are never seen in training; the viewer shows the first of them.

## Your own data

Export one row per frame and describe it in a config (see `configs/flag.yml`):

```yaml
data:
  data_file: bin/flag_simulation.npy   # (frames x 3c) or (frames, c, 3) vertex positions
  external_file: bin/flag_wind.npy     # (frames x e) external state (wind, ball position, joints, ...)
  faces_file: bin/flag.obj             # optional, lets the viewer shade the mesh
  dt: 0.016666666666666666
```

then `pdm run train-flag` (or `pdm run train configs/flag.yml`).
Several independent simulations can be stored as one `neural_physics.data.SimulationDataset` (.npz) with
`segments`; training windows and the alpha/beta fit then never cross an episode boundary and the test
split is done by episode.

### Run artifacts: model and basis are a pair

A run directory holds two files that only make sense together: `pca.npz` (the basis) and `model.pt` /
`checkpoints/*.pt` (whose every tensor is sized by `pca.n_components`). Nothing in the maths ties them
together, so:

* the basis is written **before the first epoch**, which keeps an interrupted run usable;
* a mismatched pair -- typically a basis left over from a run with a different `n_components` -- is
  reported by file name and size instead of failing deep inside the roll-out;
* `pdm run train <config> --eval-only --refit-basis` re-creates a basis that is missing or stale (valid
  only if the dataset and split are unchanged);
* changing `pca.n_components` in a config does **not** change an existing run: `--eval-only` and
  `pages-build` use the artifacts on disk and warn that the config asks for something else, until you retrain.

## Publishing trained models on GitHub Pages

The repo can serve a small showcase site (landing page + the interactive viewers) from GitHub Pages:

```bash
pdm run publish-model runs/cloth models/cloth   # snapshot model.pt, pca.npz, summary and training history (~0.5 MB)
pdm run pages-build                             # models/* -> site/  (re-evaluates each model on its held-out episode)
pdm run pages-serve                             # preview at http://localhost:8000
git add models configs/pages.yml && git commit -m "publish cloth model"
```

`.github/workflows/pages.yml` runs the same `pages-build` on every push that touches `models/`, `configs/` or the
package and deploys `site/` — the synthetic dataset is regenerated (and cached) on the runner, so only the small
snapshot in `models/` is committed. One-time setup: repository **Settings → Pages → Build and deployment →
Source: GitHub Actions**; the site then appears at `https://<user>.github.io/<repo>/`. Add more models (e.g. the
real flag) as entries in `configs/pages.yml`; for data that cannot be regenerated on the runner, either commit a
small held-out episode as a `SimulationDataset` or build locally and push `site/` to a `gh-pages` branch instead.

## Layout

```
neural_physics/
  core_math/pca.py                PCA (SVD or covariance, frame subsampling, reconstruction error, save/load)
  data/dataset.py                 SimulationDataset: positions, external state, faces, segments, dt
  data/cloth_sim.py               PBD cloth-on-a-pole generator with a wind process (synthetic training data)
  data/point_mass.py              point mass under gravity (smoke-test data)
  cli.py                          console scripts: np-datagen-cloth, np-datagen-gravity, np-train
  utils/data_preprocess.py        alpha/beta least squares (Eq. 2), segment-aware window sampling
  train/subspace_neural_physics.py  SubSpaceNeuralNetwork (Phi) and SubspaceNeuralPhysics (Eq. 1 + 3,
                                  normalisation, clipping, batched differentiable roll-out, losses)
  train/trainer.py                Algorithm 1 with mini-batches, lr decay, held-out evaluation, checkpoints
  runtime/rollout.py              auto-regressive simulation, decoding, error metrics
  runtime/export.py, viewer.html  self-contained interactive HTML viewer (three.js vendored)
  pipeline.py                     data -> PCA -> model -> training -> runtime, driven by a YAML config
  pages.py                        snapshot trained runs into models/ and build the GitHub Pages site
Examples/                         train_cloth.py, train_flag.py, train_simple_gravity.py  (pdm run train-*)
configs/                          cloth.yml, flag.yml, gravity.yml, pages.yml
models/                           published model snapshots shown on the Pages site
pyproject.toml, pdm.lock          PDM project: dependencies, scripts (`pdm run --list`), build (pdm-backend)
tox.ini                           flake8 rules for editors (the CI gate passes them on the command line)
```

`pdm run train-gravity` (`Examples/train_simple_gravity.py`) is a smoke test only: a point mass under gravity is solved exactly
by alpha = beta = 1, so the network has nothing to learn there.
