from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename,shape", [
    ("eta_julia_fast.csv", (100, 100)),
    ("u_julia_fast.csv", (100, 101)),
    ("v_julia_fast.csv", (101, 100)),
    ("energy_julia_fast.csv", (43200,)),
])
def test_saved_julia_data_shape_and_finiteness(filename, shape):
    data = np.loadtxt(ROOT / filename, delimiter=",")
    assert data.shape == shape
    assert np.isfinite(data).all()
    if filename.startswith("energy"):
        assert np.all(data >= 0)
    elif filename.startswith("u_"):
        np.testing.assert_array_equal(data[:, [0, -1]], 0.0)
    elif filename.startswith("v_"):
        np.testing.assert_array_equal(data[[0, -1], :], 0.0)


def test_originals_and_recorded_repairs_are_preserved():
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "verify_preservation.py"),
         "--allow-excluded-missing"], capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
