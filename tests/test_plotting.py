"""Headless execution of the core plotting calls used in the reproduction guide."""


def test_core_experiment_figures_are_written_in_isolated_directory(tmp_path, numpy_control):
    import matplotlib.pyplot as plt
    from analytical import run_task_c
    from diagnostics import task_d3_difference_fields, task_d3_difference_energy
    from parameters import L
    from plotting import (plot_analytical_eta, plot_energy, plot_error_fields,
                          plot_eta_snapshot, plot_grid_refinement, plot_gyre_model)
    from task_g5_sensitivity import run_g5_experiments, plot_g5_sensitivity

    difference = task_d3_difference_fields(numpy_control)
    error = {"E_error": task_d3_difference_energy(numpy_control, difference)}
    try:
        plot_analytical_eta(run_task_c(Nx=8, Ny=6))
        plot_gyre_model(numpy_control, "smoke")
        plot_energy(numpy_control, title_suffix="smoke")
        plot_error_fields(numpy_control, difference, "smoke")
        plot_grid_refinement(numpy_control, numpy_control, error, error)
        plot_eta_snapshot(numpy_control["eta"], L, days=600/86400, time=10,
                          title="Short plotting check")
        plot_g5_sensitivity(run_g5_experiments(Nx=8, Ny=6, days=120/86400, dt=10))
    finally:
        plt.close("all")
    outputs = list(tmp_path.glob("*.png"))
    assert len(outputs) == 7
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in outputs)
    assert all(path.stat().st_size > 1000 for path in outputs)
