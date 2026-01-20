# Examples — Canonical Equivariance Demo

This folder contains the minimal, reproducible demo for equivariance detection and recovery.

What varies
- `n` (problem size): number of points for the dihedral action (default 12)
- `noise` (rupture amount): non‑equivariant perturbation added to the operator (default 0.10)
- `seed` (random seed): reproducibility (default 0)
- optional fourth arg: CSV output path
- optional fifth arg: PNG output path (requires Plots)

What it measures
- Equivariance error: average of ‖g·(W x) − W (g·x)‖ over a small set of group elements and basis vectors.
- Prints:
  - `err_base` for the projected (equivariant) base operator
  - `err_rupt` after rupture
  - `err_recover` after projection
  - `ratio = err_recover / err_rupt`

Run
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/equivariance_recovery.jl 12 0.10 1
```

Artifacts (optional)
```
# CSV only
julia --project=. examples/equivariance_recovery.jl 12 0.10 1 bench/equivariance_demo.csv

# CSV + PNG (requires Plots)
julia --project=. -e 'using Pkg; Pkg.add("Plots")'
julia --project=. examples/equivariance_recovery.jl 12 0.10 1 bench/equivariance_demo.csv bench/equivariance_demo.png
```

Why this matters (brief)
- Symmetry equivariance is a robust prior improving generalization and stability under shift.
- Detection + recovery gives a principled way to repair models without retraining.
- The projection is fast and certifiable (error → ~0 within numerical tolerance).
- The demo is fair and transparent: equal data/parameters/compute, reproducible artifacts.
