from diagnostics import run_gyre_model, task_d3_difference_fields, task_d3_difference_energy
from plotting import (
    plot_analytical_eta,
    plot_gyre_model,
    plot_error_fields,
    plot_energy,
    plot_grid_refinement,
    plot_eta_snapshot,
)
from task_f import run_task_f
from parameters import L
from analytical import run_task_c
from task_g2_jax import run_gyre_model_jax
from task_g2_jax import benchmark_taskD
from task_g4_rk4 import compare_rk4_fb, stability_sweep, benchmark_rk4_fb
from task_g5_sensitivity import run_g5_experiments, plot_g5_sensitivity
from task_g3_phiflow import run_g3_phiflow, plot_g3_phiflow
from task_g7_julia import compare_julia_numpy


################
# Task C: analytical solution


task_c_initial = run_task_c(Nx=100, Ny=100, eta0=0.0)
eta0_task_c = task_c_initial["eta_ref"]

task_c = run_task_c(Nx=100, Ny=100, eta0=eta0_task_c)

print("Task C eta0:", eta0_task_c)

plot_analytical_eta(task_c)


##############
# Tasks D and E: Eulerian model


coarse_one_day = run_gyre_model(Nx=100, Ny=100, days=1, lower_dt=30.0)
coarse_steady = run_gyre_model(Nx=100, Ny=100, days=200, lower_dt=30.0)
fine_steady = run_gyre_model(Nx=200, Ny=200, days=200, lower_dt=15.0)

d3_coarse = task_d3_difference_fields(coarse_steady)
d3_fine = task_d3_difference_fields(fine_steady)

E_d3_coarse = task_d3_difference_energy(coarse_steady, d3_coarse)
E_d3_fine = task_d3_difference_energy(fine_steady, d3_fine)

print("Task D3 error energy")
print(f"100x100 E_error = {E_d3_coarse:.6e} J")
print(f"200x200 E_error = {E_d3_fine:.6e} J")
print(f"Reduction factor = {E_d3_coarse / E_d3_fine:.3f}")

plot_gyre_model(coarse_one_day, "1 day check")
plot_gyre_model(coarse_steady, "200 day steady state")
plot_error_fields(coarse_steady, d3_coarse, "100x100 steady state")
plot_energy(coarse_steady, anomaly=False, title_suffix="200_day")

plot_grid_refinement(
    coarse_steady,
    fine_steady,
    {"E_error": E_d3_coarse},
    {"E_error": E_d3_fine},
)


######################
# Task F: semi-Lagrangian model

sl_1day = run_task_f(days=1, dt=10.0)
sl_5day = run_task_f(days=5, dt=10.0)

plot_eta_snapshot(
    sl_1day["eta"],
    L,
    days=1,
    time=10.0,
    title="SL 1 day, dt = 10 s",
)

plot_eta_snapshot(
    sl_5day["eta"],
    L,
    days=5,
    time=10.0,
    title="SL 5 day, dt = 10 s",
)


############################
# Task G2

jax_result = run_gyre_model_jax(
    Nx=100,
    Ny=100,
    days=200,
    lower_dt=30.0
)

bench_g2 = benchmark_taskD(Nx=100, Ny=100, days=200, dt=30.0)

###########################
# Task G4

g4_compare = compare_rk4_fb(days=1, dt=30.0)
g4_benchmark = benchmark_rk4_fb()

print("G4 RK4 vs FB eta diff:", g4_compare["eta_diff"])
print("G4 FB runtime:", g4_benchmark["fb_time"])
print("G4 RK4 runtime:", g4_benchmark["rk4_time"])
print("G4 RK4/FB runtime ratio:", g4_benchmark["runtime_ratio"])


#########################
# Task G5

g5 = run_g5_experiments(Nx=100, Ny=100, days=30, dt=30.0)

print("G5 diagnostics:", g5["diagnostics"])

plot_g5_sensitivity(g5)



##########################
# Task G3

g3 = run_g3_phiflow()

print("G3 NumPy runtime:", g3["numpy_runtime"])
print("G3 PhiFlow runtime:", g3["phiflow_runtime"])

plot_g3_phiflow(g3)

######################
# Task G7

g7 = compare_julia_numpy(
    eta_file="eta_julia_fast.csv",
    u_file="u_julia_fast.csv",
    v_file="v_julia_fast.csv",
    energy_file="energy_julia_fast.csv",
)

print("G7 Julia speedup:", g7["speedup"])