"""Exercise the real driver with stand-ins, without launching full experiments.

Numerical backend agreement is tested separately. This test verifies imports,
return-value contracts, driver parameters and reaching the last task.
"""

from pathlib import Path
import runpy
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def test_main_driver_imports_and_result_contracts(monkeypatch, capsys):
    calls = {}

    def routine(name, result=None):
        def call(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            return result
        return call

    def module(name, **members):
        replacement = ModuleType(name)
        replacement.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, replacement)

    module("diagnostics",
           run_gyre_model=routine("baseline", {}),
           task_d3_difference_fields=routine("errors", {}),
           task_d3_difference_energy=routine("error_energy", 1.0))
    plot_names = ("plot_analytical_eta", "plot_gyre_model", "plot_error_fields",
                  "plot_energy", "plot_grid_refinement", "plot_eta_snapshot")
    module("plotting", **{name: routine(name) for name in plot_names})
    module("task_f", run_task_f=routine("semi_lagrangian", {"eta": None}))
    module("parameters", L=1e6)
    module("analytical", run_task_c=routine("analytical", {"eta_ref": 0.125}))
    # A similarly named package elsewhere on the machine must not mask the bug.
    monkeypatch.setitem(sys.modules, "stommel_exact_solution", None)
    module("task_g2_jax", run_gyre_model_jax=routine("jax", {}),
           benchmark_taskD=routine("jax_benchmark", {}))
    module("task_g4_rk4", compare_rk4_fb=routine("rk4_comparison", {"eta_diff": 0.0}),
           stability_sweep=routine("stability_sweep", []),
           benchmark_rk4_fb=routine("rk4_benchmark", {
               "fb_time": 1.25, "rk4_time": 2.5, "runtime_ratio": 2.0}))
    module("task_g5_sensitivity", run_g5_experiments=routine("sensitivity", {"diagnostics": {}}),
           plot_g5_sensitivity=routine("plot_sensitivity"))
    module("task_g3_phiflow", run_g3_phiflow=routine("phiflow", {
        "numpy_runtime": 0.1, "phiflow_runtime": 0.2}),
        plot_g3_phiflow=routine("plot_phiflow"))
    module("task_g7_julia", compare_julia_numpy=routine("julia_comparison", {"speedup": 2.0}))

    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")

    output = capsys.readouterr().out
    assert "G4 FB runtime: 1.25" in output
    assert "G4 RK4 runtime: 2.5" in output
    assert "G7 Julia speedup: 2.0" in output
    assert [kwargs for _, kwargs in calls["baseline"]] == [
        dict(Nx=100, Ny=100, days=1, lower_dt=30.0),
        dict(Nx=100, Ny=100, days=200, lower_dt=30.0),
        dict(Nx=200, Ny=200, days=200, lower_dt=15.0),
    ]
    assert calls["analytical"][1][1]["eta0"] == 0.125
    assert calls["rk4_comparison"][0][1] == dict(days=1, dt=30.0)
    assert len(calls["julia_comparison"]) == 1
    assert calls["julia_comparison"][0][1] == {
        "eta_file": "eta_julia_fast.csv", "u_file": "u_julia_fast.csv",
        "v_file": "v_julia_fast.csv", "energy_file": "energy_julia_fast.csv",
    }
