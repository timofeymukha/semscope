"""Field calculator: expression parsing, evaluation and spectrally exact derivatives."""

import numpy as np
import pytest

import semview
from semview import calc, synthetic
from semview.calc import ExpressionError


@pytest.fixture
def tg():
    return synthetic.taylor_green(synthetic.wavy_box(8, 4, 10, amplitude=0.05), k=np.pi)


def test_arithmetic_and_functions(tg):
    u, v = tg["u"], tg["v"]
    assert np.allclose(tg["sqrt(u^2 + v^2)"], np.hypot(u, v))
    assert np.allclose(tg.evaluate("2 * u - v / 2 + 1"), 2 * u - v / 2 + 1)
    assert np.allclose(tg["max(u, v) + min(u, 0) - abs(v) + atan2(v, u)"], np.maximum(u, v) + np.minimum(u, 0) - np.abs(v) + np.arctan2(v, u))
    assert np.allclose(tg["pi * exp(-x) + e"], np.pi * np.exp(-tg.x) + np.e)
    assert tg["2"].shape == tg.x.shape and np.all(tg["2"] == 2.0)
    assert np.allclose(tg["-u"], -u) and np.allclose(tg["u ** 2"], u**2)


def test_derivatives_are_spectrally_exact(tg):
    k = np.pi
    x, y = tg.x, tg.y
    vort = tg["dx(v) - dy(u)"]
    assert np.max(np.abs(vort - 2 * k * np.sin(k * x) * np.sin(k * y))) < 1e-4
    assert np.max(np.abs(tg["dx(u) + dy(v)"])) < 1e-4
    # polynomial data on straight elements: derivatives of any order are exact to round-off
    box = synthetic.box(3, 2, 7, xlim=(0, 3), ylim=(0, 1))
    synthetic.add_analytic_field(box, "f", lambda x, y: x**3 * y**2)
    assert np.max(np.abs(box["dx(f, 2)"] - 6 * box.x * box.y**2)) < 1e-9
    assert np.max(np.abs(box["dx(dy(f))"] - 6 * box.x**2 * box.y)) < 1e-9
    assert np.max(np.abs(box["dy(f, 3)"])) < 1e-8
    assert np.max(np.abs(box["dx(2.5)"])) == 0  # derivative of a constant


def test_definitions(tg):
    assert tg.available == ["u", "v", "p"]
    tg.define("speed", "sqrt(u^2 + v^2)")
    tg.define("ke", "0.5 * speed^2")
    assert tg.available == ["u", "v", "p", "speed", "ke"] and "ke" in tg and "nope" not in tg
    assert np.allclose(tg["ke"], 0.5 * (tg["u"] ** 2 + tg["v"] ** 2))
    tg.define("speed", "2 * sqrt(u^2 + v^2)")  # redefinition invalidates dependents
    assert np.allclose(tg["ke"], 2 * (tg["u"] ** 2 + tg["v"] ** 2))
    with pytest.raises(ExpressionError, match="used by"):
        tg.undefine("speed")
    tg.undefine("ke")
    tg.undefine("speed")
    assert tg.available == ["u", "v", "p"]
    with pytest.raises(KeyError):
        tg["nope"]


def test_rejected_expressions(tg):
    for bad in ["u +", "u & v", "foo(u)", "dx(u, 0)", "dx(u, 2, 3)", "q + 1", "u[0]", "__import__('os')", "u.T", "lambda: 1", "", "sqrt(u, v)"]:
        with pytest.raises(ExpressionError):
            tg.evaluate(bad)
    with pytest.raises(ExpressionError, match="unknown field"):
        tg.define("a", "b + 1")
    tg.define("a", "u")
    tg.define("b", "a")
    with pytest.raises(ExpressionError, match="circular"):
        tg.define("a", "b + 1")
    for name in ["u", "dx", "pi", "x", "1abc", "bad name", ""]:
        with pytest.raises(ExpressionError):
            tg.define(name, "u")


def test_free_names_and_help():
    assert calc.free_names("dx(u) + sqrt(v) * pi - q") == {"u", "v", "q"}
    h = calc.syntax_help()
    assert "sqrt" in h["functions"] and "atan2" in h["binary_functions"] and h["coordinates"] == ["x", "y"]


def test_dataset_definitions_apply_to_all_steps(tmp_path, tg):
    path = str(tmp_path / "tg0.f00000")
    tg.write(path)
    ds = semview.open(path)
    ds.define("w", "dx(v) - dy(u)")
    d = ds[0]
    assert d.available == ["u", "v", "p", "w"]
    assert np.max(np.abs(d["w"] - tg["dx(v) - dy(u)"])) < 1e-4
    assert ds[0].gather(ds.comm).expressions is ds.expressions
    assert ds.mesh.defined == ["w"]
    ds.undefine("w")
    assert ds[0].available == ["u", "v", "p"]
