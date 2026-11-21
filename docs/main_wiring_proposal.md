# Main driver: exact proposed repair

Status: awaiting the owner's approval. This proposal changes three lines in
`main.py`; no numerical functions, experiment parameters or ordering change.

## 1. Analytical implementation import

`from stommel_exact_solution import run_task_c` raises `ModuleNotFoundError`
because the project has no `stommel_exact_solution.py`. The existing function
is defined in `analytical.py`. Importing it from that module calls the same
analytical implementation already used by the project diagnostics.

## 2. RK4 benchmark result keys

`benchmark_rk4_fb` returns `fb_time` and `rk4_time`. The driver instead requests
`fb_runtime` and `rk4_runtime`, which raises `KeyError` after the expensive
benchmark finishes. Change the two lookups to the returned keys. This changes
only how existing timing results are printed; no timing procedure is revised.

## Exact patch

```diff
--- a/main.py
+++ b/main.py
@@
-from stommel_exact_solution import run_task_c
+from analytical import run_task_c
@@
-print("G4 FB runtime:", g4_benchmark["fb_runtime"])
-print("G4 RK4 runtime:", g4_benchmark["rk4_runtime"])
+print("G4 FB runtime:", g4_benchmark["fb_time"])
+print("G4 RK4 runtime:", g4_benchmark["rk4_time"])
```

## Preservation and verification

Before applying an approved patch, compare the file with its original manifest
hash and retain a byte-for-byte `preservation/originals/main.py` backup. Record
the repaired file's hash in `preservation/repairs.json` without replacing the
original snapshot. Preserve original line endings. Check that the exact diff
contains only these three edits.

Add an orchestration regression test using lightweight backend stand-ins to
exercise the driver wiring without executing all long experiments. That test
does not validate optional numerical backends; run their real small-grid tests
separately when installed. Rerun the full available suite after approved edits.

## Requirements retained by this minimal patch

- The driver imports JAX and PhiFlow eagerly. Both must be installed to run all
  tasks; fixing the analytical import alone does not make a core-only environment
  sufficient.
- Importing `main.py` executes the whole sequence. A main guard or task-selection
  CLI is a separate usability proposal.
- Plots are written to the current directory. Run in a new directory so saved
  research figures are not replaced.
- G7 expects the four Julia CSV filenames in that same current directory. Run
  the Julia exporter there first, or copy the preserved reference CSVs there and
  label them historical. The main driver does not invoke Julia itself.
- The driver still uses the comparison helper's historical default Julia timing.
  Do not describe its printed speedup as a fresh Julia benchmark. The dedicated
  G7 recipe accepts an explicitly measured runtime.

Changing these requirements, benchmarking methodology, parameters, output paths
or scientific algorithms is outside this three-line proposal.
