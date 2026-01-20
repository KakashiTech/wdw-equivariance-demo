# wdw-equivariance-demo

A minimal, reproducible instrument to measure and recover symmetry equivariance in linear operators.

What this is
- A tiny, focused demo you can run in under a minute.
- It measures equivariance error, induces a rupture, then restores equivariance by projection.

What this is not
- Not the full WDW system. No claims about physics or grand theories.
- No extra modules beyond what the demo needs.

Quickstart
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/equivariance_recovery.jl 12 0.10 1
```

Optional artifacts
- CSV summary:
  ```bash
  julia --project=. examples/equivariance_recovery.jl 12 0.10 1 bench/equivariance_demo.csv
  ```
- PNG bar plot (requires Plots):
  ```bash
  julia --project=. -e 'using Pkg; Pkg.add("Plots")'
  julia --project=. examples/equivariance_recovery.jl 12 0.10 1 bench/equivariance_demo.csv bench/equivariance_demo.png
  ```

Run the minimal test
```
julia --project=. -q test/test_demo_equivariance_recovery.jl
```

Why this matters (brief)
- Symmetry equivariance is a robust prior that improves generalization and stability.
- Detection + recovery shows a principled way to repair models without retraining.
- The projection is fast and certifiable (error → ~0 within numerical tolerance).
- The demo is fair and transparent: equal data/parameters/compute, reproducible artifacts.
