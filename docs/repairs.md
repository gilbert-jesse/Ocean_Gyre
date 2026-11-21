# Minimal repairs and remaining proposals

The original-file manifest was completed and verified before any source edits.
The consequences of the following two repairs were explained before editing;
the user subsequently instructed continuation with approved fixes. The source
files remain in their original locations. Original bytes are retained in
`preservation/originals/`, with before/after hashes in `preservation/repairs.json`.

## Applied: shared drag parameter

```diff
--- a/diagnostics.py
+++ b/diagnostics.py
@@
-from parameters import L, H, g, rho, f0, beta, tau0
+from parameters import L, H, g, gamma, rho, f0, beta, tau0
```

`parameters.py` already defines `gamma = 1.0e-6` (inverse seconds). The baseline
solver already includes `-gamma * dt * u` and `-gamma * dt * v`. Its missing
import caused `NameError` at the first active timestep. G5, JAX, RK4 and the
Julia implementation already use this same coefficient. Importing it restores
the intended linear drag; no new damping term or coefficient is introduced.
This does not establish long-term stability or validate historical results.

## Applied: RK4 baseline import

```diff
--- a/task_g4_rk4.py
+++ b/task_g4_rk4.py
@@
-from diagnostic import run_gyre_model
+from diagnostics import run_gyre_model
```

The existing module is `diagnostics.py`; no `diagnostic.py` exists. This repair
enables the module to import and its forward-backward comparison to run. RK4
stages, boundary conditions, forcing, time-step selection and spatial operators
are unchanged. Before repair, the focused tests reproduced `NameError` and
`ModuleNotFoundError`; see [validation](validation.md) for the final outcomes.

## Applied: combined driver wiring

After the user's explicit approval, the following exact edits were applied to
`main.py`. Its original bytes are preserved at
`preservation/originals/main.py`; original and repaired SHA-256 values are
recorded in `preservation/repairs.json`.

```diff
-from stommel_exact_solution import run_task_c
+from analytical import run_task_c
-print("G4 FB runtime:", g4_benchmark["fb_runtime"])
-print("G4 RK4 runtime:", g4_benchmark["rk4_runtime"])
+print("G4 FB runtime:", g4_benchmark["fb_time"])
+print("G4 RK4 runtime:", g4_benchmark["rk4_time"])
```

These resolve a missing module and align dictionary access with the keys
returned by `benchmark_rk4_fb`. They do not change solver calculations. The
driver also imports all optional backends and executes long experiments at
module scope. A main guard, explicit experiment selection, and isolated output
directories are proposed future usability changes, not implemented here.

No scientific algorithms, parameter values, benchmark methodology, saved
results, reports or plots were modified in this phase.
