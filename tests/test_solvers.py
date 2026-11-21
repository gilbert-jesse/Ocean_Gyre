import importlib

import numpy as np
import pytest


SOLVERS = [
    ("diagnostics", "run_gyre_model"),
    ("task_g5_sensitivity", "run_gyre_model_g5"),
    ("task_g4_rk4", "run_gyre_model_rk4"),
]


@pytest.mark.parametrize("module_name,function_name", SOLVERS)
def test_linear_solver_boundaries_mass_and_nonnegative_energy(
    module_name, function_name, assert_state,
):
    module = importlib.import_module(module_name)
    result = getattr(module, function_name)(Nx=8, Ny=6, days=600 / 86400, lower_dt=10.0)
    assert_state(result)
    # Linear continuity with closed boundaries conserves mean eta. Wind forcing
    # and drag mean total energy is NOT a conserved quantity in these runs.
    assert abs(result["eta"].mean()) < 1e-13
    assert result["time_days"][-1] == pytest.approx(600 / 86400)


@pytest.mark.parametrize("module_name,function_name", SOLVERS)
def test_unforced_rest_is_stationary(module_name, function_name, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "tau0", 0.0)
    result = getattr(module, function_name)(Nx=8, Ny=6, days=120 / 86400, lower_dt=10.0)
    for key in ("eta", "u", "v", "energy"):
        np.testing.assert_array_equal(result[key], 0.0)


def test_baseline_matches_unperturbed_g5_and_repeats(numpy_control):
    from diagnostics import run_gyre_model
    kwargs = dict(Nx=8, Ny=6, days=600 / 86400, lower_dt=10.0)
    first = run_gyre_model(**kwargs)
    second = run_gyre_model(**kwargs)
    for key in ("eta", "u", "v", "energy", "time_days"):
        np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_allclose(first[key], numpy_control[key], rtol=1e-13, atol=1e-14)


def test_g5_seeded_repeatability_and_seed_dependence(assert_state):
    from task_g5_sensitivity import run_gyre_model_g5
    kwargs = dict(Nx=8, Ny=6, days=600 / 86400, lower_dt=10.0, stochastic_amp=1e-5)
    first = run_gyre_model_g5(**kwargs, seed=123)
    same = run_gyre_model_g5(**kwargs, seed=123)
    different = run_gyre_model_g5(**kwargs, seed=124)
    assert_state(first)
    for key in ("eta", "u", "v", "energy"):
        np.testing.assert_array_equal(first[key], same[key])
    assert np.max(np.abs(first["eta"] - different["eta"])) > 0
    assert abs(first["eta"].mean()) < 1e-13


def test_g5_conserves_nonzero_mean_without_stochastic_forcing():
    from diagnostics import gaussian_eta_bump
    from task_g5_sensitivity import run_gyre_model_g5
    initial = gaussian_eta_bump(8, 6) + 0.125
    saved = initial.copy()
    result = run_gyre_model_g5(Nx=8, Ny=6, days=600 / 86400, lower_dt=10.0,
                               eta_init=initial)
    np.testing.assert_array_equal(initial, saved)
    assert abs(result["eta"].mean() - initial.mean()) < 1e-13


def test_g5_small_ensemble_has_finite_nonnegative_spread():
    from task_g5_sensitivity import run_g5_experiments
    result = run_g5_experiments(Nx=8, Ny=6, days=120 / 86400, dt=10.0)
    assert np.isfinite(result["eta_spread"]).all()
    assert np.all(result["eta_spread"] >= 0)
    assert all(np.isfinite(value) for value in result["diagnostics"].values())


def test_rk4_and_forward_backward_converge_towards_each_other():
    from diagnostics import run_gyre_model
    from task_g4_rk4 import run_gyre_model_rk4
    differences = []
    for dt in (4.0, 2.0):
        kwargs = dict(Nx=8, Ny=6, days=1200 / 86400, lower_dt=dt)
        fb = run_gyre_model(**kwargs)
        rk4 = run_gyre_model_rk4(**kwargs)
        differences.append(np.linalg.norm(fb["eta"] - rk4["eta"]))
    # A time-refinement consistency check, not a claim of RK4 order for the
    # coupled spatial discretisation or a long-integration stability guarantee.
    assert differences[0] > 0
    assert differences[1] < 0.75 * differences[0]


def test_semi_lagrangian_short_run_is_finite_and_repeatable(assert_state):
    from task_f import run_task_f
    kwargs = dict(Nx=8, Ny=6, days=120 / 86400, dt=10.0)
    first, second = run_task_f(**kwargs), run_task_f(**kwargs)
    assert_state(first)
    assert first["stable"]
    for key in ("eta", "u", "v"):
        np.testing.assert_array_equal(first[key], second[key])
    # This interpolating nonlinear method is not assumed mass-conservative.
