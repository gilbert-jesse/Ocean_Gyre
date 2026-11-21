# Phase 2 validation

## Final automated suite

**31 passed, 0 skipped, 0 failed** using Python 3.13.9, NumPy 2.3.5,
Matplotlib 3.10.6, pytest 8.4.2, JAX/JAXLIB 0.12.1, PhiFlow 3.4.0,
PhiML 1.16.1 and Julia 1.10.11. The recorded run took 27.27 seconds; this is
test-suite elapsed time, not a scientific benchmark.

Executed from the project root with the existing Python installation and
`JULIA` set to the installed Julia executable:

```console
python -B -m pytest -q -p no:cacheprovider --basetemp=tmp/phase2-full --junitxml=tmp/phase2-full.xml
```

For a normal rerun use `python -m pytest -q`, which allocates its own temporary
directory. If reproducing the custom command, ensure `tmp/` exists and select
a new `--basetemp` name. Pytest can clear an existing custom base directory;
never point it at research outputs. XML logs and temporary exports are ignored
by Git because their paths contain local machine information and are not
reference research outputs.

| Component | Status | Scope |
| --- | --- | --- |
| Analytical solution | Passed | Finiteness, rectangular grid, no-normal-flow walls, elevation gauge, approximate zero divergence |
| Staggered-grid operators | Passed | Linear interpolation/gradients and integrated divergence with closed boundaries |
| NumPy baseline | Passed | Short run, boundary conditions, mass balance, nonnegative energy, stationary unforced rest, repeatability |
| G5 | Passed | Baseline agreement, seeded repeatability and seed dependence, nonzero-mean conservation without noise, small ensemble |
| RK4 | Passed | Short run, boundaries, mass balance, unforced rest and improved agreement with FB as dt decreases |
| Semi-Lagrangian | Passed | Stationary uniform unforced state and short finite repeatable run; no general mass-conservation assertion |
| Julia | Passed | 12 native assertions plus five field/history comparisons with NumPy G5 at the same parameters |
| Historical CSVs | Passed | Four expected shapes, finite data, velocity boundaries and nonnegative energy |
| Preservation | Passed | Unchanged originals plus all three recorded repairs and original backups |
| JAX | Passed | Installed JAX/JAXLIB 0.12.1; small-grid agreement with NumPy |
| PhiFlow | Passed | Installed PhiFlow 3.4.0/PhiML 1.16.1; small-grid divergence agreement |

The linear checks use a non-square 8x6 grid with 120- or 600-second runs.
The timestep-refinement comparison uses 1200 seconds and dt=4 then 2 seconds.
Native Julia checks run for 600 seconds at dt=10 s. These are simulated
durations, not wall-clock times. Full historical experiments were not rerun.

## Regression evidence before repair

After resolving a preliminary test-runner temporary-directory setup issue,
the same three linear-solver invariant cases gave **2 failed, 1 passed**:

- Baseline: `NameError` for the missing `gamma` binding.
- RK4: `ModuleNotFoundError` for `diagnostic`.
- G5 control: passed.

The first two source fixes addressed those import failures. A later approved
three-line `main.py` repair fixed its analytical import and reported timing keys.
Exact original bytes, original/repaired hashes and change explanations are
preserved. Tests do not monkeypatch the solver imports or mark failures as
expected; the main driver test uses stand-ins to avoid launching long runs.

## Final repository and documentation checks

- The original manifest's SHA-256 still matches its creation-time value.
- Preservation verification passes: 32 originals unchanged, three recorded
  repaired files matching their new hashes, and three original backups matching
  their original hashes. No original is missing.
- All 21 Python files (source, tests, verifier and original backups) compile.
- Comparing the original and repaired abstract syntax trees after removing
  imports finds no changes to any other statements or numerical functions.
- Pytest TOML configuration and dependency-file include paths validate.
- All local Markdown links resolve; all seven documented Python reproduction
  snippets compile. Syntax validation does not mean full experiments were run.
- Read-only Git inspection confirms all four CSVs and thirteen reference PNGs
  remain publication candidates. The two PDFs, C-grid photo, trailing-dot empty
  archive and local `tmp/` test outputs are ignored. No files are staged.

## Not established by passing tests

The combined `main.py` full experiment sequence, 200-day grid-refinement
reproduction, long stability sweeps and historical timing claims have not been
reproduced. Its driver wiring has a stand-in smoke test. A finite short-run
solution is not validation of every scientific claim in the reports.

No remote CI was run and no GitHub authentication, repository creation, commit
or push was attempted. The tests can be run locally before manual publication.

## Approved main-driver wiring repair

After approval, exactly three lines in `main.py` were changed: the analytical
helper import now targets `analytical`, and the printed FB/RK4 timings now read
the returned `fb_time`/`rk4_time` keys. The byte-identical original is retained
in `preservation/originals/main.py`; original and repaired SHA-256 hashes are
recorded in `preservation/repairs.json`. No experiment settings or calculations
changed.

The driver smoke test runs `main.py` through `runpy` with stand-in experiment
modules. It exercises every task call, the analytical import and the printed
timing keys without running full integrations. It validates driver wiring, not
the scientific backends; their numerical checks remain separate tests.
