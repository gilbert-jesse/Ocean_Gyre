# Experiment reproduction guide

This guide describes how to exercise the existing implementations. It does
not claim that the full historical experiments have been reproduced. Start
with the automated short checks and read [known issues](known_issues.md).

## Isolate all new outputs

The plotting functions and Julia script write to the current directory. Use a
new directory for each selected experiment and keep the original results intact.
Example PowerShell setup from the repository root, after installing the core
environment:

```powershell
$repo = (Get-Location).Path
$pythonExe = Join-Path $repo '.venv\Scripts\python.exe'
$run = Join-Path $repo ('results\runs\run-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $run | Out-Null
$env:PYTHONPATH = $repo
$env:MPLBACKEND = 'Agg'
Push-Location $run
```

Run a selected snippet below, then execute `Pop-Location`. For scripted use,
put the selected commands inside `try { ... } finally { Pop-Location }`.
Use a new directory before another run because some filenames are fixed.
`Agg` saves plots without opening GUI windows. Record the command, source
revision or hashes, dependencies, parameters, seed, device and wall-clock
timings alongside the new outputs. All Python snippets use the original
functions; no alternate numerical implementations are introduced here.

## C: analytical reference

```powershell
@'
from analytical import run_task_c
from plotting import plot_analytical_eta
initial = run_task_c(Nx=100, Ny=100, eta0=0.0)
result = run_task_c(Nx=100, Ny=100, eta0=initial['eta_ref'])
print('Analytical eta reference:', initial['eta_ref'])
plot_analytical_eta(result)
'@ | & $pythonExe -B -
```

The Task C helper samples endpoints with `indexing="ij"`. The diagnostic
comparison functions instead evaluate the analytical fields at the numerical
staggered-grid coordinates. Do not compare array indices between these grids
without accounting for this difference and the elevation reference level.

## D/E: baseline, energy and grid refinement

The following uses the original full-run configuration and can be expensive.
The imported drag parameter is now repaired. A short initial check can replace
`Nx=100, Ny=100, days=1` with `Nx=8, Ny=6, days=600/86400` and `lower_dt=10.0`.

```powershell
@'
import numpy as np
from diagnostics import run_gyre_model, task_d3_difference_fields, task_d3_difference_energy
from plotting import plot_gyre_model, plot_energy, plot_error_fields, plot_grid_refinement
one_day = run_gyre_model(Nx=100, Ny=100, days=1, lower_dt=30.0)
coarse = run_gyre_model(Nx=100, Ny=100, days=200, lower_dt=30.0)
fine = run_gyre_model(Nx=200, Ny=200, days=200, lower_dt=15.0)
coarse_error = task_d3_difference_fields(coarse)
fine_error = task_d3_difference_fields(fine)
ec = task_d3_difference_energy(coarse, coarse_error)
ef = task_d3_difference_energy(fine, fine_error)
print('Error energies (J):', ec, ef, 'reduction:', ec / ef)
plot_gyre_model(one_day, '1 day check')
plot_gyre_model(coarse, '200 day steady state')
plot_error_fields(coarse, coarse_error, '100x100 steady state')
plot_energy(coarse, title_suffix='200_day')
plot_grid_refinement(coarse, fine, {'E_error': ec}, {'E_error': ef})
np.savez('coarse_fields.npz', **coarse)
np.savez('fine_fields.npz', **fine)
'@ | & $pythonExe -B -
```

This prints freshly computed quantities rather than copying values from the
report. Error energy is not total model energy. The two diagnostic entry points
use different reference-level constructions; record which one was used.

## F: nonlinear semi-Lagrangian experiment

```powershell
@'
from task_f import run_task_f
from plotting import plot_eta_snapshot
from parameters import L
for days in (1, 5):
    result = run_task_f(Nx=100, Ny=100, days=days, dt=10.0)
    print('Requested days:', days, 'stable:', result['stable'],
          'reported elapsed days:', result['time_reached_days'])
    plot_eta_snapshot(result['eta'], L, days=days, time=10.0,
                      title=f'SL requested {days} days; stable={result["stable"]}')
'@ | & $pythonExe -B -
```

The historical report describes instability at larger timesteps and longer
durations. Preserve unsuccessful runs and their stop diagnostics. A stopped
run is not a full-duration solution. The original early-stop elapsed-time
field uses the zero-based failing step and may underreport elapsed time by
one timestep; this behaviour is documented, not changed.

## G2: optional JAX implementation

In a separately validated JAX environment, with `$pythonExe` pointing to it:

```powershell
@'
from task_g2_jax import benchmark_taskD
result = benchmark_taskD(Nx=100, Ny=100, days=200, dt=30.0)
for key in ('numpy_runtime', 'jax_first_runtime', 'jax_compiled_runtime',
            'speedup', 'eta_diff', 'u_diff', 'v_diff', 'energy_diff'):
    print(key, result[key])
'@ | & $pythonExe -B -
```

The existing wrapper runs the JAX solver twice internally; its outer benchmark
therefore does not time one identical unit of work per reported NumPy/JAX
runtime. Treat these as the existing experiment's timing definitions and record
them explicitly. A revised benchmark methodology requires a separate change.

## G3: optional PhiFlow operator experiment

With PhiFlow installed in the selected environment:

```powershell
& $pythonExe -B (Join-Path $repo 'task_g3_phiflow.py')
```

This creates a centred scalar field and staggered velocity field, computes
gradient/divergence operators, and times 100 divergence evaluations. The
PhiFlow timing includes conversion to NumPy. It does not integrate a gyre.

## G4: RK4 and stability comparisons

```powershell
@'
from task_g4_rk4 import compare_rk4_fb, stability_sweep, benchmark_rk4_fb
result = compare_rk4_fb(Nx=100, Ny=100, days=1, dt=30.0)
print({key: result[key] for key in ('eta_diff', 'u_diff', 'v_diff')})
# Long optional sweeps; unstable runs may emit floating-point warnings:
# print(stability_sweep(method='rk4', Nx=100, Ny=100, days=30))
# print(stability_sweep(method='fb', Nx=100, Ny=100, days=30))
# benchmark = benchmark_rk4_fb(Nx=100, Ny=100)
# print(benchmark['fb_time'], benchmark['rk4_time'], benchmark['runtime_ratio'])
'@ | & $pythonExe -B -
```

The existing runtime comparison uses FB dt=60 s and RK4 dt=100 s over 30 days.
It compares these configurations, not equal-step computational cost. The sweep
catches all exceptions and labels them nonfinite, so inspect unexpected failures
before attributing them to numerical instability.

## G5: initial conditions and stochastic ensemble

```powershell
@'
import numpy as np
from task_g5_sensitivity import run_g5_experiments, plot_g5_sensitivity
result = run_g5_experiments(Nx=100, Ny=100, days=30, dt=30.0)
print(result['diagnostics'])
plot_g5_sensitivity(result)
np.savez('g5_summary_fields.npz', eta_mean=result['eta_mean'],
         eta_spread=result['eta_spread'], eta_diff_nearby=result['eta_diff_nearby'])
'@ | & $pythonExe -B -
```

Initial Gaussian amplitudes are 0.001, 0.01 and 0.05 m. Nearby initial-condition
noise has amplitude 1e-6 with seed 123. The ten stochastic members use seeds
0 through 9 and amplitude 1e-5 per forward-backward cycle. Record timestep as
well as amplitude: the perturbations are not scaled by the square root of dt.

## G7: Julia and NumPy comparison

From the isolated run directory, using the tested Julia installation:

```powershell
julia --startup-file=no (Join-Path $repo 'G7_GYRE.jl')
```

The script runs 100x100, 30 days, dt=30 s twice, then writes four CSVs here.
The function's default dt=10 s differs from these explicit top-level calls.
Preserve both timing outputs; the second is the warmed run. For a comparison
with the newly generated CSVs, supply the actual measured runtime instead of
the Python helper's historical default of 4.72 seconds:

```powershell
$measuredJuliaSeconds = Read-Host 'Enter the measured second Julia runtime in seconds'
@'
import sys
from task_g7_julia import compare_julia_numpy
result = compare_julia_numpy(Nx=100, Ny=100, days=30, dt=30.0,
                             julia_runtime=float(sys.argv[1]))
print(result)
'@ | & $pythonExe -B - $measuredJuliaSeconds
```

To compare against the historical CSVs instead, pass their absolute paths via
`eta_file`, `u_file`, `v_file` and `energy_file`; label that comparison historical
and do not associate it with a newly measured Julia runtime. CSVs do not embed
their parameters, timing or environment. The saved 43,200 energy values are
consistent with 30 days at two 30-second substeps per recorded cycle, but this
does not independently prove their original provenance.
