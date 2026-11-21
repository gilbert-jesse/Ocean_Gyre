import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.jax
def test_jax_matches_numpy_forward_backward(numpy_control, assert_state):
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX not installed; optional backend untested")
    from task_g2_jax import run_gyre_model_jax
    result = run_gyre_model_jax(Nx=8, Ny=6, days=600 / 86400, lower_dt=10.0)
    assert_state(result)
    for key in ("eta", "u", "v", "energy", "time_days"):
        np.testing.assert_allclose(result[key], numpy_control[key], rtol=1e-12, atol=1e-14)


@pytest.mark.phiflow
def test_phiflow_divergence_matches_independent_staggered_difference():
    if importlib.util.find_spec("phi") is None:
        pytest.skip("PhiFlow not installed; optional backend untested")
    from task_g3_phiflow import run_g3_phiflow
    from parameters import L
    nx, ny = 8, 6
    result = run_g3_phiflow(Nx=nx, Ny=ny)
    dx, dy = L / nx, L / ny
    xu, yu = np.meshgrid(np.arange(nx + 1) * dx, (np.arange(ny) + 0.5) * dy)
    xv, yv = np.meshgrid((np.arange(nx) + 0.5) * dx, np.arange(ny + 1) * dy)
    u = 0.02 * np.sin(np.pi * xu / L) * np.sin(np.pi * yu / L)
    v = 0.02 * np.sin(np.pi * xv / L) * np.sin(np.pi * yv / L)
    expected = np.diff(u, axis=1) / dx + np.diff(v, axis=0) / dy
    assert result["eta_np"].shape == (ny, nx)
    assert np.isfinite(result["eta_np"]).all()
    np.testing.assert_allclose(result["div_phi_np"], expected, rtol=1e-5, atol=1e-12)


@pytest.mark.julia
def test_julia_solver_and_numpy_agree(tmp_path, numpy_control):
    executable = os.environ.get("JULIA") or shutil.which("julia")
    if not executable:
        pytest.skip("Julia executable unavailable; set JULIA or add Julia to PATH")
    env = os.environ.copy()
    env["JULIA_DEPOT_PATH"] = str(tmp_path / "julia_depot")
    completed = subprocess.run(
        [executable, "--startup-file=no", "--history-file=no", "--compiled-modules=no",
         str(ROOT / "tests" / "julia_smoke.jl"), str(ROOT / "G7_GYRE.jl"), str(tmp_path)],
        capture_output=True, text=True, env=env, timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for key in ("eta", "u", "v", "energy", "time_days"):
        actual = np.loadtxt(tmp_path / (key + ".csv"), delimiter=",")
        np.testing.assert_allclose(actual, numpy_control[key], rtol=1e-12, atol=1e-14)
