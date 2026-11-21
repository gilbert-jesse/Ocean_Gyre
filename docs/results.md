# Preserved results and provenance

The original-file manifest contains 35 files totalling 6,327,784 bytes before
Phase 2 additions. The four CSVs, thirteen PNG figures, two PDFs and two JPG
diagrams were not regenerated, overwritten, moved or deleted. Three source
files received import/reporting-only repairs; their exact original copies and
the before/after hashes are recorded separately.

## Julia CSVs

All four original CSVs contain finite numeric values and no headers. Shapes
were checked by loading the data, not inferred solely from filenames.

| File | Shape | Meaning from the exporting code |
| --- | --- | --- |
| `eta_julia_fast.csv` | 100 x 100 | Elevation at cell centres, m |
| `u_julia_fast.csv` | 100 x 101 | Zonal velocity at x-faces, m/s |
| `v_julia_fast.csv` | 101 x 100 | Meridional velocity at y-faces, m/s |
| `energy_julia_fast.csv` | 43,200 values | Total perturbation energy, J, one sample per two-substep cycle |

Rows correspond to y and columns to x in the solver. `G7_GYRE.jl` currently
exports these filenames after explicit 100x100, 30-day runs at dt=30 s. The
energy count agrees with that configuration. The CSVs have no embedded run
metadata or associated original environment lock, so this correspondence is
not proof of the exact generator revision, runtime or hardware. There is no
saved time-vector CSV. Do not assign a time axis without recording the assumed
dt and cycle convention.

## Figure inventory

Paths below are under `Figures/`; original names are retained.

| File | Content / provenance limitation |
| --- | --- |
| `analytical solution.png` | Analytical elevation; generated filename differs from current plotting helper |
| `gyre_model_1_day_check.png` | Labelled one-day baseline fields and transects |
| `gyre_model_200_day_steady_state.png` | Labelled 200-day baseline fields and transects |
| `gyre_model_100x100.png` | Related model diagnostic figure; generic label does not establish duration |
| `energy_plot.png` | Energy history with a 60-day displayed axis |
| `energy_plot_energy.png` | Energy history with a 200-day displayed axis |
| `error_fields_100x100.png` | Numerical-minus-analytical fields; retain as its own historical artifact |
| `error_fields_steady_state.png` | Another error-field figure with different values/labels |
| `fine_coarse_grid (1).png` | Coarse/fine energy evolution and error-energy comparison |
| `SL_PLOTS.png` | Semi-Lagrangian snapshots including an unstable/checkerboard result |
| `Task_G4_RK4_eta_after_1_day__dt_=_10_s.png` | RK4 elevation; filename and image say dt=10 s, report caption says dt=30 s |
| `Sensitivity_to_initial_conditions (1).png` | Initial-condition and stochastic ensemble comparison |
| `PhiFlow_grid.png` | Scalar field and divergence-operator visualisation |

The PNGs with software metadata identify Matplotlib 3.10.0. This is historical
metadata, not evidence that all dependency versions or scripts match the
current files. There are no original byte-identical duplicates, and visually
similar outputs should not be discarded as redundant.

## Reports and diagrams

- `Final_Report.pdf`: ten-page research report with methods, results, references
  and appendices. Its text includes a name and student identifier.
- `REVISED_TASK A AND B.pdf`: two-page revised theoretical analysis. Its extracted
  text still includes personal identifiers even where they may not be visible.
- `C-GRID.jpg`: handwritten staggered-grid diagram with a visible identifier.
- `FLOWCHART.jpg`: handwritten algorithm flowchart; retained as a publication
  candidate subject to the owner's final review.

The two PDFs and C-grid photo are excluded by exact `.gitignore` rules; their
original files remain local. No redacted copies have been created. The report's
original LaTeX source, bibliography source and a full set of raw numerical
outputs for every experiment are not present in this project.

## Distinguish historical claims from new checks

Historical report numbers are not copied into a benchmark leaderboard here.
The preparation run checks small-grid correctness properties and preservation;
it does not recreate the 200-day convergence study, published timings, full
stability sweeps or every figure. Newly generated experiments belong under
`results/runs/` with parameters, seeds, dependency versions, hardware, source
hashes and timing definitions recorded alongside them.
