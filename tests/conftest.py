"""Keep test outputs away from the preserved research results."""

import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))


@pytest.fixture
def assert_state():
    def check(result, nx=8, ny=6):
        assert result["eta"].shape == (ny, nx)
        assert result["u"].shape == (ny, nx + 1)
        assert result["v"].shape == (ny + 1, nx)
        for key in ("eta", "u", "v"):
            assert np.isfinite(result[key]).all(), key
        np.testing.assert_array_equal(result["u"][:, [0, -1]], 0.0)
        np.testing.assert_array_equal(result["v"][[0, -1], :], 0.0)
        if "energy" in result:
            assert result["energy"].size > 0
            assert np.isfinite(result["energy"]).all()
            assert np.all(result["energy"] >= 0.0)
    return check


@pytest.fixture
def numpy_control():
    from task_g5_sensitivity import run_gyre_model_g5
    return run_gyre_model_g5(Nx=8, Ny=6, days=600 / 86400, lower_dt=10.0)
