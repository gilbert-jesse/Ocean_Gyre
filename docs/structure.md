# Current layout and proposed organisation

Source files have not been moved or renamed. Python modules and the Julia
source remain at the root so that existing relative imports and documented
entry points remain identifiable.

| Current item | Responsibility |
| --- | --- |
| `parameters.py` | Shared physical constants |
| `analytical.py` | Analytical Stommel/Musgrave reference fields, Task C |
| `diagnostics.py` | NumPy forward-backward solver, energy and error diagnostics |
| `plotting.py` | Common plotting and filename construction |
| `semi_lagrangian.py` | Bilinear interpolation, C-grid operators and nonlinear timestep |
| `task_f.py` | Semi-Lagrangian experiment and instability detection |
| `task_g2_jax.py` | JAX solver and benchmarking |
| `task_g3_phiflow.py` | PhiFlow grid/operator experiment, not a full time-integrating gyre model |
| `task_g4_rk4.py` | RK4, stability sweeps and forward-backward comparison |
| `task_g5_sensitivity.py` | Initial-condition and stochastic ensemble experiments |
| `G7_GYRE.jl` | Julia solver, timings and CSV export |
| `task_g7_julia.py` | NumPy comparison with Julia CSVs |
| `main.py` | Combined experiment driver; remaining wiring issues documented |
| `Figures/` and four root CSVs | Preserved historical results |
| PDFs and JPGs | Report, revised theory and handwritten diagrams |
| `Project_TWO.` | Empty gzip/tar archive; retained locally and ignored |
| `docs/` | Added scientific, reproduction and publication documentation |
| `requirements/`, `pyproject.toml` | Added dependencies and pytest configuration |
| `tests/` | Added Python and Julia checks |
| `scripts/verify_preservation.py` | Read-only checksum verification |
| `preservation/` | Original manifest, two original source backups and repair records |

## Proposed later layout

```text
Project_TWO/
  README.md
  pyproject.toml
  requirements/
  src/gyre/
    parameters.py
    analytical.py
    diagnostics.py
    plotting.py
    semi_lagrangian.py
  experiments/
    numpy/          # Task F and sensitivity entry points
    jax/            # Identifiable Task G2 implementation
    phiflow/        # Identifiable Task G3 operator experiment
    rk4/            # Identifiable Task G4 implementation
    comparisons/    # Task G7 Python comparison
  julia/
    G7_GYRE.jl
  scripts/          # Explicit experiment orchestration
  tests/
  docs/
    reports/        # Reviewed sharing copies only
    diagrams/
  results/
    reference/      # Original CSVs and figures with provenance
    runs/           # Ignored new outputs
  preservation/
  local_only/       # Ignored original personal material
```

This is a proposal, not the current tree. Such a move requires reviewed imports,
packaging, output-path changes and regression tests. Keep full independent
implementations identifiable instead of prematurely merging their timesteps.

## Safe cleanup proposals

No original files were byte-identical duplicates in the initial review.
`energy_plot.png` and `energy_plot_energy.png` show different durations.
Similar gyre/error plots and filenames containing `(1)` are not evidence that
the results are disposable. Preserve them until provenance establishes their
relationship. The G5 plotting function is duplicated exactly in `plotting.py`
and `task_g5_sensitivity.py`; consolidation is proposed, not applied.

`Project_TWO.` contains an empty tar archive and its trailing dot prevents
ordinary Windows tooling from opening it reliably. It is ignored and hashed,
not deleted. No cleanup or source relocation has been performed.
