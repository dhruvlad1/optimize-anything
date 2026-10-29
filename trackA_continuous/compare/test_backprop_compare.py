"""Tool-independent tests for the backprop comparison (Track A, part 2).

Both AI tools get the same task; this file judges them equally.
Expected files in this folder: backprop_antigravity.py, backprop_claude.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
CANDIDATES = ["backprop_antigravity.py", "backprop_claude.py"]
KEYS = ["W1", "b1", "W2", "b2"]


def load(filename):
    path = HERE / filename
    if not path.exists():
        pytest.skip(f"{filename} not created yet")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_data(seed=0, n=7, d=3, k=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.integers(0, k, size=n)


def reference_loss(params, X, y, l2):
    """Independent stable loss: mean softmax cross-entropy + 0.5*l2*(|W1|^2+|W2|^2)."""
    h = np.tanh(X @ params["W1"] + params["b1"])
    z = h @ params["W2"] + params["b2"]
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    ce = -logp[np.arange(len(y)), y].mean()
    return ce + 0.5 * l2 * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))


def numeric_grad(mod, params, X, y, l2, key, eps=1e-5):
    g = np.zeros_like(params[key])
    for idx in np.ndindex(*params[key].shape):
        plus = {k: v.copy() for k, v in params.items()}
        minus = {k: v.copy() for k, v in params.items()}
        plus[key][idx] += eps
        minus[key][idx] -= eps
        g[idx] = (mod.loss_and_grads(plus, X, y, l2=l2)[0]
                  - mod.loss_and_grads(minus, X, y, l2=l2)[0]) / (2 * eps)
    return g


@pytest.mark.parametrize("filename", CANDIDATES)
@pytest.mark.parametrize("l2", [0.0, 0.1])
def test_gradient_matches_finite_differences(filename, l2):
    mod = load(filename)
    X, y = make_data()
    params = mod.init_params(3, 4, 3, seed=1)
    _, grads = mod.loss_and_grads(params, X, y, l2=l2)
    for key in KEYS:
        num = numeric_grad(mod, params, X, y, l2, key)
        rel = np.linalg.norm(num - grads[key]) / max(
            1e-8, np.linalg.norm(num) + np.linalg.norm(grads[key]))
        assert rel < 1e-6, f"{key}: relative error {rel:.2e}"


@pytest.mark.parametrize("filename", CANDIDATES)
@pytest.mark.parametrize("l2", [0.0, 0.1])
def test_loss_value_matches_reference(filename, l2):
    mod = load(filename)
    X, y = make_data()
    params = mod.init_params(3, 4, 3, seed=1)
    loss, _ = mod.loss_and_grads(params, X, y, l2=l2)
    assert abs(loss - reference_loss(params, X, y, l2)) < 1e-10


@pytest.mark.parametrize("filename", CANDIDATES)
def test_shapes_and_keys(filename):
    mod = load(filename)
    X, y = make_data()
    params = mod.init_params(3, 4, 3, seed=1)
    shapes = {"W1": (3, 4), "b1": (4,), "W2": (4, 3), "b2": (3,)}
    assert {k: v.shape for k, v in params.items()} == shapes
    loss, grads = mod.loss_and_grads(params, X, y)
    assert np.ndim(loss) == 0
    assert {k: v.shape for k, v in grads.items()} == shapes


@pytest.mark.parametrize("filename", CANDIDATES)
def test_large_logits_stay_finite(filename):
    mod = load(filename)
    X, y = make_data()
    params = mod.init_params(3, 4, 3, seed=1)
    params["W2"] = params["W2"] * 1000.0
    loss, grads = mod.loss_and_grads(params, X, y)
    assert np.isfinite(loss)
    assert all(np.all(np.isfinite(g)) for g in grads.values())


@pytest.mark.parametrize("filename", CANDIDATES)
def test_inputs_not_modified(filename):
    mod = load(filename)
    X, y = make_data()
    params = mod.init_params(3, 4, 3, seed=1)
    before = {k: v.copy() for k, v in params.items()}
    Xb, yb = X.copy(), y.copy()
    mod.loss_and_grads(params, X, y, l2=0.1)
    assert all(np.array_equal(before[k], params[k]) for k in KEYS)
    assert np.array_equal(Xb, X) and np.array_equal(yb, y)


@pytest.mark.parametrize("filename", CANDIDATES)
def test_init_deterministic(filename):
    mod = load(filename)
    a, b, c = (mod.init_params(3, 4, 3, seed=s) for s in (5, 5, 6))
    assert all(np.array_equal(a[k], b[k]) for k in KEYS)
    assert any(not np.array_equal(a[k], c[k]) for k in KEYS)


@pytest.mark.parametrize("filename", CANDIDATES)
def test_gradient_descent_reduces_loss(filename):
    mod = load(filename)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])  # XOR: needs the hidden layer to work
    params = mod.init_params(2, 6, 2, seed=0)
    first = mod.loss_and_grads(params, X, y)[0]
    for _ in range(2000):
        _, grads = mod.loss_and_grads(params, X, y)
        for k in KEYS:
            params[k] = params[k] - 0.5 * grads[k]
    assert mod.loss_and_grads(params, X, y)[0] < 0.5 * first