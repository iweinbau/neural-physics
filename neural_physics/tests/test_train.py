import numpy as np
import torch

from neural_physics.train.subspace_neural_physics import (
    SubSpaceNeuralNetwork,
    SubspaceNeuralPhysics,
    loss_fn,
    loss_position,
    loss_velocity,
)
from neural_physics.train.trainer import TrainConfig, evaluate_short_horizon, train


def test_network_shapes_and_depth():
    batch_size, n_components, n_external = 32, 16, 3
    model = SubSpaceNeuralNetwork(num_components_X=n_components, num_components_Y=n_external, n_hidden_layers=8)
    assert model.n_linear_layers == 10                      # the paper's "10 layers"
    inputs = torch.rand(batch_size, 2 * n_components + n_external)
    assert model(inputs).shape == (batch_size, n_components)
    assert model.encode[0].out_features == round(1.5 * n_components)


def test_model_fit_linear_and_rollout_shapes(oscillator_data):
    Z, W, dt = oscillator_data
    model = SubspaceNeuralPhysics(Z.shape[0], W.shape[0], n_layers=10, dt=dt)
    assert model.net.n_linear_layers == 10
    model.fit_linear(Z, W)
    assert torch.all(model.alphas < 1.0) and torch.all(model.alphas > 0.9)
    assert torch.all(model.betas > 0.9) and torch.all(model.betas < 1.01)
    assert torch.all(model.out_scale > 0) and torch.all(model.z_max > model.z_min)
    zw = torch.tensor(Z[:, :32].T)[None].repeat(4, 1, 1)
    ww = torch.tensor(W[:, :32].T)[None].repeat(4, 1, 1)
    out = model.rollout(zw[:, 0], zw[:, 1], ww)
    assert out.shape == (4, 32, Z.shape[0])
    assert torch.equal(out[:, 0], zw[:, 0]) and torch.equal(out[:, 1], zw[:, 1])
    # clipped roll-out stays inside the training range
    clipped = model.rollout(zw[:, 0] * 50, zw[:, 1] * 50, ww, clip=True)
    assert torch.all(clipped[:, 2:] <= model.z_max + 1e-6) and torch.all(clipped[:, 2:] >= model.z_min - 1e-6)
    # alpha/beta-only roll-out does not depend on the network
    base = model.rollout(zw[:, 0], zw[:, 1], ww, use_network=False)
    manual = [zw[:, 0], zw[:, 1]]
    for i in range(2, 32):
        manual.append(model.alphas * manual[-1] + model.betas * (manual[-1] - manual[-2]))
    assert torch.allclose(base, torch.stack(manual, 1), atol=1e-5)


def test_rollout_backpropagates_through_time(oscillator_data):
    Z, W, dt = oscillator_data
    torch.manual_seed(0)
    model = SubspaceNeuralPhysics(Z.shape[0], W.shape[0], n_layers=4, dt=dt).fit_linear(Z, W)
    zw = torch.tensor(Z[:, :32].T)[None]
    ww = torch.tensor(W[:, :32].T)[None]
    out = model.rollout(zw[:, 0], zw[:, 1], ww)
    loss_fn(out, zw, dt).backward()
    g_full = [p.grad.clone() for p in model.net.parameters()]
    model.net.zero_grad()
    # same roll-out with the recurrence detached: gradients must differ
    preds = [zw[:, 0], zw[:, 1]]
    for i in range(2, 32):
        preds.append(model.step(preds[-1].detach(), preds[-2].detach(), ww[:, i]))
    loss_fn(torch.stack(preds, 1), zw, dt).backward()
    g_cut = [p.grad.clone() for p in model.net.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(g_full, g_cut))
    assert all(torch.isfinite(g).all() for g in g_full)


def test_loss_fn_matches_definition():
    torch.manual_seed(0)
    z_star, z, dt = torch.rand(3, 10, 4), torch.rand(3, 10, 4), 1 / 30
    expected = loss_position(z_star[:, 2:], z[:, 2:]) + loss_velocity(z_star[:, 2:], z_star[:, 1:-1], z[:, 2:], z[:, 1:-1], dt)
    assert torch.allclose(loss_fn(z_star, z, dt), expected)
    assert loss_fn(z, z, dt).item() == 0.0


def test_train_reduces_loss_and_saves_checkpoints(oscillator_data, tmp_path):
    Z, W, dt = oscillator_data
    Ztr, Wtr, Zte, Wte = Z[:, :2400], W[:, :2400], Z[:, 2400:], W[:, 2400:]
    model = SubspaceNeuralPhysics(Z.shape[0], W.shape[0], n_layers=4, dt=dt).fit_linear(Ztr, Wtr)
    before = evaluate_short_horizon(model, torch.tensor(Zte), torch.tensor(Wte), n_windows=32)
    cfg = TrainConfig(epochs=3, steps_per_epoch=40, batch_size=16, lr=1e-3, noise_std=0.02, eval_windows=32,
                      checkpoint_dir=str(tmp_path / "ckpt"), verbose=False, device="cpu", seed=0)
    history = train(model, Ztr, Wtr, None, Zte, Wte, None, cfg)
    assert len(history) == 3 and all(np.isfinite(h["loss"]) for h in history)
    assert history[-1]["loss"] < history[0]["loss"]
    assert (tmp_path / "ckpt" / "best.pt").exists() and (tmp_path / "ckpt" / "history.json").exists()
    assert abs(history[-1]["lr"] - 1e-3 * 0.999 ** 3) < 1e-9          # decay applied once per epoch
    after = evaluate_short_horizon(model, torch.tensor(Zte), torch.tensor(Wte), n_windows=32)
    assert after["baseline"] == before["baseline"]                    # alpha/beta untouched by training
    reloaded = SubspaceNeuralPhysics.load(tmp_path / "ckpt" / "last.pt")
    zw = torch.tensor(Zte[:, :32].T)[None]
    ww = torch.tensor(Wte[:, :32].T)[None]
    assert torch.allclose(reloaded.rollout(zw[:, 0], zw[:, 1], ww), model.rollout(zw[:, 0], zw[:, 1], ww), atol=1e-6)


def test_linear_skip_is_zero_initialised_and_reference_beats_alpha_beta(oscillator_data):
    Z, W, dt = oscillator_data
    torch.manual_seed(0)
    plain = SubspaceNeuralPhysics(Z.shape[0], W.shape[0], n_layers=10, dt=dt).fit_linear(Z, W)
    torch.manual_seed(0)
    skip = SubspaceNeuralPhysics(Z.shape[0], W.shape[0], n_layers=10, dt=dt, linear_skip=True).fit_linear(Z, W)
    assert skip.net.n_linear_layers == 10 and skip.net.skip is not None
    x = torch.rand(5, plain.net.input_size)
    assert torch.allclose(plain.net(x), skip.net(x))               # zero-initialised skip changes nothing at init
    cfg = skip.config()
    assert cfg["linear_skip"] is True
    from neural_physics.train.trainer import linear_residual_reference
    ref = linear_residual_reference(skip, Z[:, :2400], W[:, :2400], None, Z[:, 2400:], W[:, 2400:], None, n_windows=32)
    assert 0 < ref["r2"] <= 1.0 and ref["linear"] < ref["baseline"]
