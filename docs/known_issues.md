# Known issues and limits

## Repaired execution blockers

The missing baseline `gamma` import, incorrect RK4 `diagnostic` import and
three-line `main.py` import/timing-key wiring were repaired. See [the exact changes](repairs.md) and
[validation results](validation.md). No discretisation or physical parameter
was changed.

## Remaining combined-driver blockers

The analytical import and RK4 timing keys in `main.py` now match the existing
modules. The driver still eagerly imports optional JAX and PhiFlow, has no main
guard, and launches substantial work at module scope. Its wiring is smoke-tested
with stand-ins; a full combined run remains expensive and environment-dependent.

## Scientific and reporting provenance

- The final report lists a CFL bound around 353.5 s for its stated 100x100
  configuration. The source formula and revised Task A/B analysis give about
  70.7 s for dx=dy=10 km and sqrt(g*H)=100 m/s. Retain the original report and
  record a correction separately; do not silently replace it.
- The saved RK4 image says dt=10 s; its report caption and current one-day
  comparison use dt=30 s. The exact provenance of that figure is unresolved.
- `run_task_c` uses endpoint samples and `indexing="ij"`; numerical solvers
  use cell centres/faces and `indexing="xy"`. Diagnostic comparisons correctly
  evaluate analytical fields at each relevant numerical grid. Reusing raw
  arrays from Task C as if the grids were identical would be inappropriate.
- No original dependency lock, complete run metadata or source revision is
  supplied with the historical results. The current checked environment is
  documented separately, including the older Matplotlib version in image
  metadata.

## Behaviour retained without algorithm changes

- Wind forcing and drag exchange energy with the system. Tests check finite,
  nonnegative energy, not constant total energy.
- Closed-boundary linear continuity conserves mean elevation to roundoff.
  The interpolating nonlinear semi-Lagrangian method is not assumed exactly
  mass-conservative. Its original clipping, depth floor and source correction
  remain unchanged.
- The report describes semi-Lagrangian checkerboard instability for longer
  runs. Passing short tests is not proof of long-term stability. The early-stop
  elapsed-time field uses a zero-based failing-step index and can underreport
  elapsed time by one dt. This behaviour is documented but not repaired.
- Forward-backward methods store one energy/time sample for every two dt
  updates; RK4 stores one per dt. Duration is truncated to an integer number
  of cycles/steps. Compare at matched reached times, not just array indices.
- G5 stochastic forcing is added once per cycle and has no sqrt(dt) scaling.
  Changing timestep changes its interpretation; amplitudes alone do not define
  equivalent stochastic experiments.
- JAX's benchmark wraps a function that performs both first and warmed runs.
  Its outer timing cannot be interpreted as a single integration cost.
- The Julia comparison helper defaults to a historical 4.72-second runtime;
  supply a newly measured value only when comparing corresponding new runs.
- The RK4 timing comparison uses different stable timesteps for the two
  methods. The stability sweep catches any exception, so runtime errors can
  be misclassified as numerical failure.
- Existing scripts save plots/CSVs to the working directory with fixed or
  title-derived names. Always use a fresh output directory.

Addressing these scientific or methodological issues requires a separately
reviewed change and appropriate new validation. They were not altered to make
the preservation and short-run tests pass.
