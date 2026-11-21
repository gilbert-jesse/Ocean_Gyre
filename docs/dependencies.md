# Dependencies and environment provenance

## Verified environment

The short validation checks use the existing Windows Anaconda installation:

| Component | Observed version | Role |
| --- | --- | --- |
| Python | 3.13.9 | Python experiments and test runner |
| NumPy | 2.3.5 | Arrays, numerical operators and CSV loading |
| Matplotlib | 3.10.6 | Plotting; imported by several experiment modules |
| pytest | 8.4.2 | Added automated tests |
| Julia | 1.10.11 | Independent Julia solver |
| DelimitedFiles | 1.9.1 | Julia CSV output |
| Printf | Bundled with Julia | Imported by original Julia source |
| Test | Bundled with Julia | New Julia checks |

This is a validation environment, not a recovered original environment.
Existing PNG metadata records Matplotlib 3.10.0. The original Python, NumPy,
JAX and PhiFlow versions and original hardware were not supplied. Do not infer
them from today's installations. No GPU or speedup claim is made by the tests.

## Python installation

From the project root with a configured Python 3.13 interpreter:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/test.txt
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS/Linux, the environment interpreter is `.venv/bin/python`. The source
is not yet an installable package; `pyproject.toml` only configures pytest. Do
not use `pip install -e .` for this snapshot. `requirements.txt` installs only
the core direct dependencies; `requirements/test.txt` includes pytest.

The exact core pins match the inspected environment. They are not a complete
transitive lock, and a clean virtual-environment installation has not been
validated in this phase. No packages were installed or upgraded in the
scientific environment during Phase 2.

## Optional backends

| Backend | Source | Dependency file | Validation status |
| --- | --- | --- | --- |
| JAX / JAXLIB | `task_g2_jax.py` | `requirements/jax.txt` | JAX 0.12.1 / JAXLIB 0.12.1; small-grid consistency check passed |
| PhiFlow / PhiML | `task_g3_phiflow.py` | `requirements/phiflow.txt` | PhiFlow 3.4.0 / PhiML 1.16.1; small-grid divergence check passed |

The optional dependency files deliberately have no version pins. The versions
above were resolved and tested in a separate environment against the pinned
core dependencies; this is a recorded validation combination, not an original
experiment lock. The `phiflow` distribution supplies `phi.flow`; do not
substitute an unrelated package named `phi`.

```powershell
# In an appropriate optional environment, one backend at a time:
python -m pip install -r requirements/jax.txt
python -m pytest -q -m jax
# Or:
python -m pip install -r requirements/phiflow.txt
python -m pytest -q -m phiflow
```

JAX explicitly enables 64-bit arithmetic in the existing source. A GPU is not
required by the test, and no accelerator installation instructions are assumed.
Record Python/package versions, device, OS, CPU/GPU and thread settings with
future benchmark results. Review environment exports for local paths or private
package URLs before including them in Git.

## Julia

Use the tested Julia 1.10.11 executable. The source imports `Printf` and
`DelimitedFiles`, both available in the inspected installation; there are no
third-party numerical solver packages in this implementation. A Julia
`Project.toml`/`Manifest.toml` has not been fabricated from incomplete provenance.

The Python test runner discovers Julia on PATH, or through the `JULIA`
environment variable:

```powershell
$env:JULIA = 'C:\path\to\julia.exe'
python -m pytest -q -m julia
```

The harness disables startup/history files and confines its depot and exported
comparison data to pytest's temporary directory. It reads the unchanged Julia
solver definitions and skips the original script's top-level benchmarks.
