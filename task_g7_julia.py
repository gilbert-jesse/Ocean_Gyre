import time
import numpy as np

from diagnostics import run_gyre_model


def compare_julia_numpy(
    eta_file="eta_julia_fast.csv",
    u_file="u_julia_fast.csv",
    v_file="v_julia_fast.csv",
    energy_file="energy_julia_fast.csv",
    Nx=100,
    Ny=100,
    days=30,
    dt=30.0,
    julia_runtime=4.72,
):
    start = time.time()

    res_np = run_gyre_model(
        Nx=Nx,
        Ny=Ny,
        days=days,
        lower_dt=dt
    )

    numpy_runtime = time.time() - start

    eta_julia = np.loadtxt(eta_file, delimiter=",")
    u_julia = np.loadtxt(u_file, delimiter=",")
    v_julia = np.loadtxt(v_file, delimiter=",")
    energy_julia = np.loadtxt(energy_file, delimiter=",")

    return {
        "numpy_runtime": numpy_runtime,
        "julia_runtime": julia_runtime,
        "speedup": numpy_runtime / julia_runtime,
        "eta_diff": np.max(np.abs(res_np["eta"] - eta_julia)),
        "u_diff": np.max(np.abs(res_np["u"] - u_julia)),
        "v_diff": np.max(np.abs(res_np["v"] - v_julia)),
        "energy_diff": np.max(np.abs(res_np["energy"] - energy_julia)),
    }


if __name__ == "__main__":
    g7 = compare_julia_numpy()

    print("Python NumPy runtime:", g7["numpy_runtime"])
    print("Julia runtime:", g7["julia_runtime"])
    print("speedup:", g7["speedup"])
    print("max eta difference:", g7["eta_diff"])
    print("max u difference:", g7["u_diff"])
    print("max v difference:", g7["v_diff"])
    print("max energy difference:", g7["energy_diff"])